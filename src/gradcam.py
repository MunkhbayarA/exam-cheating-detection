"""Grad-CAM for the R(2+1)D classifier: overlay a heatmap of where the model
looks when predicting a clip's behavior.

Usage:
    python gradcam.py <video>            # explain one clip
    python gradcam.py                    # auto-pick an example from the test split
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset import CLASSES  # noqa: E402
from inference import frames_to_tensor, load_model, sample_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def compute_cam(model, x, cls):
    """Grad-CAM map over the last conv block. x: (1, C, T, H, W). Returns (T, H, W)."""
    acts, grads = {}, {}
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    try:
        logits = model(x)
        if cls is None:
            cls = int(logits.argmax(1))
        model.zero_grad()
        logits[0, cls].backward()
        A, G = acts["v"][0], grads["v"][0]          # (C, T', H', W')
        weights = G.mean(dim=(1, 2, 3))             # channel importance
        cam = torch.relu((weights[:, None, None, None] * A).sum(0))
        cam = cam / (cam.max() + 1e-6)
        t, h, w = x.shape[2:]
        cam = F.interpolate(cam[None, None], size=(t, h, w), mode="trilinear",
                            align_corners=False)[0, 0]
        return cam.detach().cpu().numpy(), cls
    finally:
        h1.remove()
        h2.remove()


def pick_example():
    """First test-split clip (a 'copying' one if present) as a default example."""
    df = pd.read_csv(ROOT / "manifest_splits.csv")
    te = df[df.split == "test"]
    pref = te[te.label == "copying"]
    row = (pref if len(pref) else te).iloc[0]
    return row["path"], row["label"]


def main():
    REPORTS.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        video, true_label = sys.argv[1], None
    else:
        video, true_label = pick_example()
        print(f"auto-picked example ({true_label}): {video}")

    model, device = load_model()
    frames = sample_frames(video)
    x = frames_to_tensor(frames).unsqueeze(0).to(device)
    cam, cls = compute_cam(model, x, cls=None)

    idxs = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(2 * len(idxs), 2.5))
    for ax, t in zip(axes, idxs):
        heat = cv2.applyColorMap((cam[t] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * frames[t] + 0.45 * heat).astype(np.uint8)
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"frame {t}", fontsize=8)
    title = f"Grad-CAM — predicted: {CLASSES[cls]}"
    if true_label:
        title += f"  (true: {true_label})"
    fig.suptitle(title)
    fig.tight_layout()
    out = REPORTS / "gradcam.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"predicted {CLASSES[cls]} — wrote {out}")


if __name__ == "__main__":
    main()
