"""Gradio demo: upload an exam clip and see the model's behavior prediction.

Run locally:
    python app/app.py
Then open the printed local URL. Also the entry point for the HuggingFace Space.
"""
import sys
from pathlib import Path

# Works in both layouts: app/app.py inside the repo (src/ one level up) and
# app.py at the root of a HuggingFace Space (src/ alongside).
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "src").is_dir() else _HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from dataset import CLASSES  # noqa: E402
from inference import load_model, predict  # noqa: E402

_MODEL = None  # loaded lazily on first request so importing this module stays cheap
_DEVICE = None

TITLE = "🎓 Exam Behavior Classifier"
DESCRIPTION = (
    "Upload a short clip of a single student and the model predicts one of five "
    "behaviors: **normal, copying, gesture, mobile, notes**. It samples 16 frames, "
    "center-crops them, and runs an R(2+1)D-18 video CNN fine-tuned from Kinetics-400."
)
DISCLAIMER = (
    "⚠️ **Research & education demo only — not a proctoring system.** Trained on a small "
    "dataset, it can be wrong and may carry dataset bias. Treat any output as a *suggestion "
    "to review*, never as evidence of cheating. See the repo's *Limitations & Responsible "
    "Use* section."
)


def _ensure_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _MODEL, _DEVICE = load_model()
    return _MODEL, _DEVICE


def classify(video_path):
    if not video_path:
        return {}, []
    model, device = _ensure_model()
    probs, frames = predict(video_path, model=model, device=device)
    gallery = frames[::2]  # show every other sampled frame — a light preview strip
    return probs, gallery


def build_demo():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                inp = gr.Video(label="Clip of one student", sources=["upload"])
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out_label = gr.Label(num_top_classes=5, label="Predicted behavior")
                out_gallery = gr.Gallery(
                    label="Frames the model saw", columns=8, height=140, object_fit="cover"
                )
        btn.click(classify, inputs=inp, outputs=[out_label, out_gallery])
        gr.Markdown(f"Classes: {', '.join(CLASSES)}")
    return demo


if __name__ == "__main__":
    build_demo().launch()
