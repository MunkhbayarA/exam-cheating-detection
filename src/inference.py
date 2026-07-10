"""Reusable inference: load the trained R(2+1)D-18 and classify a video clip.

Shared by the CLI (predict.py) and the Gradio demo (app/app.py) so both use
identical preprocessing and model loading.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

from dataset import CLASSES, CROP, MEAN, N_FRAMES_OUT, STD

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "best_model.pt"


def load_model(ckpt_path=DEFAULT_CKPT, device=None):
    """Load the fine-tuned checkpoint. Returns (model, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = r2plus1d_18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, device


def sample_frames(video_path):
    """Uniformly sample N_FRAMES_OUT center-cropped RGB frames (uint8, CROP x CROP).

    Same preprocessing the model was trained on. Returns a list of numpy arrays,
    usable both as model input and as a visual preview.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read video: {video_path}")
    idxs = set(np.linspace(0, total - 1, N_FRAMES_OUT).astype(int).tolist())
    frames, pos = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pos in idxs:
            h, w = frame.shape[:2]
            side = min(h, w)
            crop = frame[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
            crop = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pos += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from: {video_path}")
    while len(frames) < N_FRAMES_OUT:  # pad short clips with the last frame
        frames.append(frames[-1])
    return frames[:N_FRAMES_OUT]


def frames_to_tensor(frames):
    """Stack sampled frames into a normalized (C, T, H, W) tensor."""
    t = torch.from_numpy(np.stack(frames)).float().div_(255)
    t = t.permute(3, 0, 1, 2)
    return (t - MEAN) / STD


@torch.no_grad()
def predict(video_path, model=None, device=None):
    """Classify one clip. Returns (probs_dict, sampled_frames).

    probs_dict maps each class name to its probability (sums to 1).
    """
    if model is None:
        model, device = load_model(device=device)
    frames = sample_frames(video_path)
    x = frames_to_tensor(frames).unsqueeze(0).to(device)
    if device == "cuda":
        with torch.autocast("cuda"):
            logits = model(x).float()
    else:
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}, frames
