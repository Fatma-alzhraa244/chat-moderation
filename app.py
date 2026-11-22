"""Real-time chat moderation demo.

    python app.py

Models are loaded from ./artifacts, or downloaded from the Hugging Face Hub
repo named in the ARTIFACTS_REPO environment variable (see README).
The default model is the quantized Transformer (DistilBERT INT8).
"""
import html
import os
import time
from pathlib import Path

import gradio as gr

from common import NAME_QBERT, load_predictors, mask_text

ART = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
REPO = os.environ.get("ARTIFACTS_REPO", "")

if not (ART / "meta.json").exists():
    if not REPO:
        raise SystemExit("No ./artifacts folder found. Set ARTIFACTS_REPO=<user>/<repo> "
                         "(Hugging Face Hub) or run train.py first. See README.")
    from huggingface_hub import snapshot_download
    ART = Path(snapshot_download(repo_id=REPO))

MODELS = load_predictors(ART)
for m in MODELS.values():          # warm-up so the first message is not slow
    m.proba("hello there")
CHOICES = list(MODELS)
DEFAULT = NAME_QBERT if NAME_QBERT in MODELS else CHOICES[-1]
LOG_HEADERS = ["#", "model", "label", "confidence", "latency (ms)"]


def render(chat):
    if not chat:
        return "<i>No messages yet - type something below.</i>"
    rows = []
    for who, shown, label, conf, ms, model in chat:
        color = "#c0392b" if label == "offensive" else "#27ae60"
        rows.append(
            f"<div style='margin:6px 0'><b>{html.escape(who)}:</b> {html.escape(shown)}<br>"
            f"<span style='color:{color}'>&#9873; {label} ({conf:.1%})</span> "
            f"<span style='color:#888'>&middot; {ms:.1f} ms &middot; {html.escape(model)}</span></div>")
    return "".join(rows)


def send(message, model_name, chat, log):
    if not message or not message.strip():
        return chat, log, "", render(chat), log
    model = MODELS[model_name]
    t0 = time.perf_counter()
    p = model.proba(message)
    ms = (time.perf_counter() - t0) * 1000
    offensive = p >= 0.5
    label = "offensive" if offensive else "clean"
    conf = p if offensive else 1 - p
    shown = mask_text(message) if offensive else message      # flagged messages are masked
    chat = chat + [("You", shown, label, conf, ms, model_name)]
    log = log + [[len(log) + 1, model_name, label, f"{conf:.3f}", f"{ms:.1f}"]]
    return chat, log, "", render(chat), log


def clear():
    return [], [], render([]), []


with gr.Blocks(title="Real-Time Chat Moderation") as demo:
    gr.Markdown("## Real-Time Chat Moderation\nType a message. Offensive messages are masked. "
                "Switch models on the right and watch the latency log.")
    chat_state, log_state = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat_view = gr.HTML(render([]))
            box = gr.Textbox(placeholder="Type a chat message and press Enter", label="Message")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(CHOICES, value=DEFAULT, label="Model")
            log_view = gr.Dataframe(headers=LOG_HEADERS, value=[], interactive=False,
                                    label="Response-time log")
    inputs = [box, model_dd, chat_state, log_state]
    outputs = [chat_state, log_state, box, chat_view, log_view]
    box.submit(send, inputs, outputs)
    send_btn.click(send, inputs, outputs)
    clear_btn.click(clear, None, [chat_state, log_state, chat_view, log_view])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE") == "1")
