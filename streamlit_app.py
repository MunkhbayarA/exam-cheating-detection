"""Streamlit demo: upload an exam clip and see the model's behavior prediction.

Live on Streamlit Community Cloud; the checkpoint is pulled from the HF model repo
(mbradiant/exam-behavior-classifier) so the 125MB file never lives in git.

Run locally:
    streamlit run streamlit_app.py
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import load_model, predict  # noqa: E402

HF_REPO = "mbradiant/exam-behavior-classifier"

st.set_page_config(page_title="Exam Behavior Classifier", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading model (first run downloads ~125MB)...")
def get_model():
    local_ckpt = ROOT / "best_model.pt"
    if local_ckpt.exists():  # local dev: use the checkpoint next to the repo
        return load_model(local_ckpt)
    from huggingface_hub import hf_hub_download
    return load_model(hf_hub_download(HF_REPO, "best_model.pt"))


st.title("🎓 Exam Behavior Classifier")
st.markdown(
    "Upload a short clip of a **single student** and an R(2+1)D-18 video CNN "
    "(fine-tuned from Kinetics-400) predicts one of five behaviors: "
    "**normal, copying, gesture, mobile, notes**."
)
st.warning(
    "**Research & education demo — not a proctoring system.** Trained on a small dataset; "
    "treat any output as a suggestion to review, never as evidence of cheating. "
    "See the [repo](https://github.com/MunkhbayarA/exam-cheating-detection) for the full "
    "limitations & responsible-use discussion.",
    icon="⚠️",
)

up = st.file_uploader("Video clip (a few seconds is enough)", type=["mp4", "mov", "avi", "mkv"])

if up is not None:
    suffix = Path(up.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp_path = tmp.name
    try:
        model, device = get_model()
        with st.spinner("Sampling frames and classifying..."):
            probs, frames = predict(tmp_path, model=model, device=device)
    except Exception as e:  # noqa: BLE001 — surface decode errors to the user
        st.error(f"Could not process this clip: {e}")
    else:
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top_class, top_p = ranked[0]
        st.subheader(f"Prediction: `{top_class}`  ({top_p:.1%})")
        for name, p in ranked:
            st.progress(min(max(p, 0.0), 1.0), text=f"{name} — {p:.1%}")

        st.caption("Frames the model saw (16 sampled uniformly, center-cropped):")
        cols = st.columns(8)
        for i, frame in enumerate(frames[::2]):
            cols[i % 8].image(frame, use_container_width=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

st.divider()
st.caption(
    "Model: [mbradiant/exam-behavior-classifier](https://huggingface.co/mbradiant/exam-behavior-classifier) · "
    "Code & pipeline: [MunkhbayarA/exam-cheating-detection](https://github.com/MunkhbayarA/exam-cheating-detection)"
)
