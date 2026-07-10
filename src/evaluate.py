"""Evaluate best checkpoint on the held-out test split."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

from dataset import CLASSES, ClipDataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    device = "cuda"
    df = pd.read_csv(ROOT / "manifest_splits.csv")
    te = df[df.split == "test"]
    loader = DataLoader(ClipDataset(te, train=False), batch_size=8, num_workers=2)

    ckpt = torch.load(ROOT / "best_model.pt", map_location=device, weights_only=True)
    model = r2plus1d_18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (val_f1 {ckpt['val_f1']:.3f})")

    ys, ps = [], []
    with torch.no_grad(), torch.autocast("cuda"):
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    print("\n" + classification_report(y, p, target_names=CLASSES, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, p)
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

    te = te.reset_index(drop=True)
    wrong = te[y != p].copy()
    if len(wrong):
        wrong["predicted"] = [CLASSES[i] for i in p[y != p]]
        print("\nmisclassified clips:")
        print(wrong[["source", "clip_id", "label", "predicted"]].to_string(index=False))


if __name__ == "__main__":
    main()
