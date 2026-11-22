"""Train and evaluate the whole model ladder on the SAME held-out test set.

    python train.py                      # defaults (needs a GPU for speed, ~10 min on a Colab T4)
    python train.py --epochs 2           # faster

Outputs (all produced by YOUR run - nothing is hard-coded):
    artifacts/            model files (upload these to Hugging Face Hub, do NOT commit them)
    results.csv / .md     the results table (task metric, size, latency)
    test_predictions.csv  every model's prediction on every test message
    failures_int8.csv     test messages the quantized Transformer got wrong
    data_stats.json, hardware.txt
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import (MAX_LEN, NAME_BERT, NAME_QBERT, NAME_RULE, NAME_TFIDF, NAME_W2V,
                    BertModel, RuleModel, TfidfModel, W2VModel, benchmark, build_lexicon,
                    embed, hardware_info, make_ffn, normalize, quantize_int8, tokenize)

SEED = 42
ART = Path("artifacts")


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_data():
    from datasets import load_dataset
    last = None
    for args in (("cardiffnlp/tweet_eval", "offensive"), ("tweet_eval", "offensive")):
        try:
            return load_dataset(*args)
        except Exception as e:  # noqa
            last = e
            print("could not load", args, "->", e)
    raise RuntimeError(f"Dataset download failed: {last}")


def size_mb(*paths):
    total = 0
    for p in paths:
        p = Path(p)
        total += sum(f.stat().st_size for f in p.parent.glob(p.name + "*") if f.is_file())
    return total / 1e6


def macro_f1(y, p):
    return f1_score(y, (np.asarray(p) >= 0.5).astype(int), average="macro")


# ---------------------------------------------------------------------------
def train_w2v_ffn(trX, trY, vaX, vaY, epochs=60):
    print("\n== Area A: Word2Vec (trained on our messages) + feedforward net (SGD + backprop)")
    w2v = Word2Vec(sentences=[tokenize(t) for t in trX], vector_size=100, window=5,
                   min_count=2, sg=1, epochs=30, workers=2, seed=SEED)
    kv = w2v.wv
    Xtr = torch.from_numpy(np.stack([embed(t, kv) for t in trX]))
    ytr = torch.tensor(trY)
    net = make_ffn(kv.vector_size)
    counts = np.bincount(trY, minlength=2)
    weights = torch.tensor(len(trY) / (2.0 * counts), dtype=torch.float32)  # class imbalance
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            loss.backward()   # backpropagation
            opt.step()        # SGD step
        f1 = macro_f1(vaY, W2VModel(kv, net).proba_batch(vaX))
        if f1 > best:
            best, best_state = f1, copy.deepcopy(net.state_dict())
        if ep % 10 == 0:
            print(f"  epoch {ep:2d}  val macro-F1 {f1:.4f}")
    net.load_state_dict(best_state)
    print(f"  best val macro-F1 {best:.4f}")
    return kv, net


def train_bert(model_name, trX, trY, vaX, vaY, epochs, lr, bs):
    print(f"\n== Area C: fine-tuning {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  training device:", device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(trX)
    total_steps = epochs * math.ceil(n / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    Xn = [normalize(t) or "." for t in trX]
    y = np.array(trY)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for k, i in enumerate(range(0, n, bs)):
            idx = perm[i:i + bs]
            enc = tok([Xn[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            labels = torch.tensor(y[idx]).to(device)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if k % 100 == 0:
                print(f"  epoch {ep + 1}/{epochs} step {k} loss {loss.item():.4f}")
        model.eval()
        f1 = macro_f1(vaY, BertModel(tok, model, "tmp", device).proba_batch(vaX))
        print(f"  epoch {ep + 1} val macro-F1 {f1:.4f}")
    return tok, model.cpu().eval()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased",
                    help="any BERT-family encoder, e.g. bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--runs", type=int, default=200, help="latency runs (must be >= 100)")
    ap.add_argument("--budget_ms", type=float, default=50.0, help="latency budget per message")
    args = ap.parse_args()
    set_seed()
    ART.mkdir(exist_ok=True)

    # ---------------- data ----------------
    ds = load_data()
    trX, trY = list(ds["train"]["text"]), list(ds["train"]["label"])
    vaX, vaY = list(ds["validation"]["text"]), list(ds["validation"]["label"])
    teX, teY = list(ds["test"]["text"]), np.array(ds["test"]["label"])
    stats = {s: {"n": len(ds[s]), "offensive_pct": round(100 * float(np.mean(ds[s]["label"])), 1)}
             for s in ("train", "validation", "test")}
    print("DATA:", json.dumps(stats))
    Path("data_stats.json").write_text(json.dumps(stats, indent=2))

    # ---------------- Area B: rule baseline + tf-idf baseline ----------------
    print("\n== Area B: edit-distance lexicon rule baseline")
    lexicon = build_lexicon(trX, trY)
    print("  lexicon size:", len(lexicon))
    (ART / "lexicon.json").write_text(json.dumps(lexicon))
    rule = RuleModel(lexicon)

    print("\n== Area B: TF-IDF + logistic regression baseline")
    best_f1, best_pipe = -1, None
    for C in (0.3, 1, 3, 10):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                                      min_df=2, sublinear_tf=True)),
            ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
        ])
        pipe.fit(trX, trY)
        f1 = macro_f1(vaY, pipe.predict_proba(vaX)[:, 1])
        print(f"  C={C}: val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_pipe = f1, pipe
    joblib.dump(best_pipe, ART / "tfidf_lr.joblib")
    tfidf = TfidfModel(best_pipe)

    # ---------------- Area A: word2vec + FFN ----------------
    kv, net = train_w2v_ffn(trX, trY, vaX, vaY)
    kv.save(str(ART / "w2v.kv"))
    torch.save(net.state_dict(), ART / "ffn.pt")
    w2v = W2VModel(kv, net)

    # ---------------- Area C: Transformer ----------------
    tok, bert = train_bert(args.model, trX, trY, vaX, vaY, args.epochs, args.lr, args.bs)
    tok.save_pretrained(ART / "tokenizer")
    bert.config.save_pretrained(ART / "bert_config")
    torch.save(bert.state_dict(), ART / "bert_fp32.pt")

    # ---------------- Area D: quantization ----------------
    print("\n== Area D: dynamic INT8 quantization")
    qbert = quantize_int8(copy.deepcopy(bert))
    torch.save(qbert.state_dict(), ART / "bert_int8.pt")
    (ART / "meta.json").write_text(json.dumps({"base_model": args.model, "max_len": MAX_LEN}))

    models = {
        NAME_RULE: (rule, size_mb(ART / "lexicon.json")),
        NAME_TFIDF: (tfidf, size_mb(ART / "tfidf_lr.joblib")),
        NAME_W2V: (w2v, size_mb(ART / "w2v.kv", ART / "ffn.pt")),
        NAME_BERT: (BertModel(tok, bert, NAME_BERT), size_mb(ART / "bert_fp32.pt")),
        NAME_QBERT: (BertModel(tok, qbert, NAME_QBERT), size_mb(ART / "bert_int8.pt")),
    }

    # ---------------- evaluation on the SAME test set ----------------
    hw = hardware_info()
    Path("hardware.txt").write_text(hw)
    print("\nHARDWARE:", hw)
    rows, preds, probs = [], {"text": teX, "label": teY}, {}
    for name, (m, mb) in models.items():
        p = m.proba_batch(teX)
        probs[name] = p
        preds[name] = (p >= 0.5).astype(int)
        mean_ms, p95_ms = benchmark(m, teX, runs=max(100, args.runs))
        rows.append({"model": name,
                     "macro_f1": round(float(macro_f1(teY, p)), 4),
                     "accuracy": round(float(accuracy_score(teY, preds[name])), 4),
                     "size_mb": round(mb, 2),
                     "latency_mean_ms": round(mean_ms, 2),
                     "latency_p95_ms": round(p95_ms, 2)})
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv("results.csv", index=False)
    md = res.to_markdown(index=False) if hasattr(res, "to_markdown") else res.to_string(index=False)
    Path("results.md").write_text(
        f"Test set: {len(teX)} messages (same for every model). "
        f"Latency: mean over {max(100, args.runs)} single-message runs.\n\nHardware: {hw}\n\n{md}\n")
    pd.DataFrame(preds).to_csv("test_predictions.csv", index=False)

    q = pd.DataFrame({"text": teX, "label": teY, "p_offensive": probs[NAME_QBERT]})
    q["pred"] = (q.p_offensive >= 0.5).astype(int)
    q[q.pred != q.label].assign(conf=lambda d: (d.p_offensive - 0.5).abs()) \
        .sort_values("conf", ascending=False).to_csv("failures_int8.csv", index=False)

    q_lat = rows[-1]["latency_mean_ms"]
    print(f"\nLatency budget {args.budget_ms} ms -> INT8 model mean {q_lat} ms:",
          "MET" if q_lat <= args.budget_ms else "NOT MET (change the budget honestly, or use a smaller model)")
    print("\nDONE. Next: upload the artifacts/ folder to Hugging Face Hub, then run app.py")


if __name__ == "__main__":
    main()
