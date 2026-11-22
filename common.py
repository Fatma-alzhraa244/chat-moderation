"""Shared code for training (train.py) and the live app (app.py).

Everything that touches text goes through normalize() so that training and
inference see exactly the same input.
"""
import json
import os
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Names used in tables and in the app (keep identical everywhere)
# ----------------------------------------------------------------------------
NAME_RULE = "Rule: edit-distance lexicon"
NAME_TFIDF = "Baseline: TF-IDF + LogReg"
NAME_W2V = "Neural: Word2Vec + FFN"
NAME_BERT = "Transformer: DistilBERT FP32"
NAME_QBERT = "Quantized: DistilBERT INT8"
MAX_LEN = 64  # max tokens fed to the Transformer

# ----------------------------------------------------------------------------
# Area B - regex normalization of noisy chat text
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # "fuuuuck" -> "fuuck"
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                      "5": "s", "7": "t", "@": "a", "$": "s"})
HAS_LETTER = re.compile(r"[a-z]")
HAS_LEET = re.compile(r"[013457@$]")


def normalize(text: str) -> str:
    """lowercase -> drop URLs/@mentions/# -> undo leetspeak -> squeeze repeats."""
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    out = []
    for tok in t.split():
        # leetspeak only inside tokens that also contain letters ("b4d" -> "bad"),
        # so plain numbers like "2026" are untouched
        if HAS_LETTER.search(tok) and HAS_LEET.search(tok):
            tok = tok.translate(LEET)
        tok = REPEAT_RE.sub(r"\1\1", tok)
        tok = re.sub(r"[^a-z0-9']", " ", tok)
        out.append(tok)
    return " ".join(" ".join(out).split())


def tokenize(text: str):
    return normalize(text).split()


# ----------------------------------------------------------------------------
# Area B - minimum edit distance + lexicon matcher (rule baseline)
# ----------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Classic Levenshtein DP (insert = delete = substitute = 1)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def build_lexicon(texts, labels, min_count=5, min_ratio=0.85, min_len=3):
    """Words that appear >= min_count times in TRAIN data and are almost
    always in offensive messages. Built from the training split only."""
    total, off = Counter(), Counter()
    for t, y in zip(texts, labels):
        for w in set(tokenize(t)):
            total[w] += 1
            off[w] += int(y == 1)
    return sorted(w for w, c in total.items()
                  if c >= min_count and len(w) >= min_len and off[w] / c >= min_ratio)


class RuleModel:
    name = NAME_RULE

    def __init__(self, lexicon):
        self.lexicon = list(lexicon)
        self.set = set(self.lexicon)

    def _hit(self, tok):
        if tok in self.set:
            return True
        if len(tok) >= 5:  # fuzzy match only for longer words (avoid false hits)
            for w in self.lexicon:
                if abs(len(w) - len(tok)) <= 1 and len(w) >= 5 and edit_distance(tok, w) <= 1:
                    return True
        return False

    def proba(self, text):
        return 0.9 if any(self._hit(t) for t in tokenize(text)) else 0.1

    def proba_batch(self, texts):
        return np.array([self.proba(t) for t in texts])


# ----------------------------------------------------------------------------
# Baseline classical model (tf-idf + logistic regression), pickled sklearn pipeline
# ----------------------------------------------------------------------------
class TfidfModel:
    name = NAME_TFIDF

    def __init__(self, pipeline):
        self.pipe = pipeline

    def proba_batch(self, texts):
        return self.pipe.predict_proba(list(texts))[:, 1]

    def proba(self, text):
        return float(self.proba_batch([text])[0])


# ----------------------------------------------------------------------------
# Area A - Word2Vec (trained by us) -> feedforward network
# ----------------------------------------------------------------------------
def make_ffn(dim=100):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))


def embed(text, kv):
    vecs = [kv[w] for w in tokenize(text) if w in kv]
    if not vecs:
        return np.zeros(kv.vector_size, dtype="float32")
    return np.mean(vecs, axis=0).astype("float32")


class W2VModel:
    name = NAME_W2V

    def __init__(self, kv, net):
        self.kv, self.net = kv, net.eval()

    def proba_batch(self, texts):
        import torch
        X = np.stack([embed(t, self.kv) for t in texts]).astype("float32")
        with torch.inference_mode():
            p = torch.softmax(self.net(torch.from_numpy(X)), dim=1)[:, 1]
        return p.numpy()

    def proba(self, text):
        return float(self.proba_batch([text])[0])


# ----------------------------------------------------------------------------
# Area C + D - Transformer (FP32) and its dynamic INT8 quantized version
# ----------------------------------------------------------------------------
class BertModel:
    def __init__(self, tok, model, name, device="cpu"):
        self.tok, self.model, self.name, self.device = tok, model.eval(), name, device

    def proba_batch(self, texts, bs=64):
        import torch
        out = []
        for i in range(0, len(texts), bs):
            chunk = [normalize(t) or "." for t in texts[i:i + bs]]
            enc = self.tok(chunk, truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def proba(self, text):
        return float(self.proba_batch([text])[0])


def quantize_int8(model):
    """Dynamic INT8 quantization of all Linear layers (runs on CPU)."""
    import torch
    from torch.ao.quantization import quantize_dynamic
    return quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


def _load_bert(art: Path, quantized: bool):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(art / "tokenizer"))
    cfg = AutoConfig.from_pretrained(str(art / "bert_config"))
    model = AutoModelForSequenceClassification.from_config(cfg)
    if quantized:
        model = quantize_int8(model)
    sd = torch.load(art / ("bert_int8.pt" if quantized else "bert_fp32.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return BertModel(tok, model, NAME_QBERT if quantized else NAME_BERT)


def load_predictors(art):
    """Load every model that exists in the artifacts folder -> {name: model}."""
    import joblib
    art = Path(art)
    models = {}
    if (art / "lexicon.json").exists():
        models[NAME_RULE] = RuleModel(json.loads((art / "lexicon.json").read_text()))
    if (art / "tfidf_lr.joblib").exists():
        models[NAME_TFIDF] = TfidfModel(joblib.load(art / "tfidf_lr.joblib"))
    if (art / "w2v.kv").exists() and (art / "ffn.pt").exists():
        import torch
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(str(art / "w2v.kv"))
        net = make_ffn(kv.vector_size)
        net.load_state_dict(torch.load(art / "ffn.pt", map_location="cpu"))
        models[NAME_W2V] = W2VModel(kv, net)
    if (art / "bert_fp32.pt").exists():
        models[NAME_BERT] = _load_bert(art, quantized=False)
    if (art / "bert_int8.pt").exists():
        models[NAME_QBERT] = _load_bert(art, quantized=True)
    return models


# ----------------------------------------------------------------------------
# UI helpers + benchmarking
# ----------------------------------------------------------------------------
def mask_text(text: str) -> str:
    return re.sub(r"[A-Za-z0-9]", "*", text)


def hardware_info() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import torch
        threads = torch.get_num_threads()
    except Exception:
        threads = "n/a"
    return f"CPU: {cpu} | logical cores: {os.cpu_count()} | torch threads: {threads} | {platform.platform()}"


def benchmark(model, texts, runs=200, warmup=10):
    """Latency of ONE message at a time (batch size 1), full path:
    normalization + features + model. Returns (mean_ms, p95_ms)."""
    for i in range(warmup):
        model.proba(texts[i % len(texts)])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        model.proba(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))
