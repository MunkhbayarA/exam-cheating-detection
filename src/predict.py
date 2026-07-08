"""Predict the behavior class of any video clip.

Usage:
    python predict.py <video1> [video2 ...]
    python predict.py "..\archive\V2\V2\s4.mp4"
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

from dataset import CLASSES, CROP, MEAN, N_FRAMES_OUT, STD

ROOT = Path(__file__).resolve().parents[1]


def load_clip(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"cannot read {video_path}")
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
    while len(frames) < N_FRAMES_OUT:
        frames.append(frames[-1])
    t = torch.from_numpy(np.stack(frames[:N_FRAMES_OUT])).float().div_(255)
    t = t.permute(3, 0, 1, 2)  # (C, T, H, W)
    return (t - MEAN) / STD


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT / "best_model.pt", map_location=device, weights_only=True)
    model = r2plus1d_18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    for p in paths:
        x = load_clip(p).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device):
            probs = torch.softmax(model(x).float(), dim=1)[0].cpu()
        top = probs.argsort(descending=True)
        print(f"\n{Path(p).name}")
        for i in top[:3]:
            bar = "#" * int(probs[i] * 30)
            print(f"  {CLASSES[i]:<8} {probs[i]:6.1%}  {bar}")


if __name__ == "__main__":
    main()
