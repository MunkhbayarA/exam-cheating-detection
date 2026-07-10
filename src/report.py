"""Generate report artifacts into reports/: confusion matrix, per-class metrics,
a metrics summary, and (if train.py logged one) the training curve.

Usage:
    python report.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader  # noqa: E402

from dataset import CLASSES, ClipDataset  # noqa: E402
from inference import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def predict_test_split():
    df = pd.read_csv(ROOT / "manifest_splits.csv")
    te = df[df.split == "test"].reset_index(drop=True)
    loader = DataLoader(ClipDataset(te, train=False), batch_size=8, num_workers=0)
    model, device = load_model()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if device == "cuda":
                with torch.autocast("cuda"):
                    out = model(x)
            else:
                out = model(x)
            ps.append(out.argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def plot_confusion(y, p):
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test split)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=140)
    plt.close(fig)


def plot_training_curve():
    hist_path = REPORTS / "history.csv"
    if not hist_path.exists():
        print("  (no history.csv — skip training curve; it appears after a fresh train.py run)")
        return
    h = pd.read_csv(hist_path)
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(h.epoch, h.train_loss, color="tab:red", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(h.epoch, h.val_f1, color="tab:blue", label="val macro-F1")
    ax2.plot(h.epoch, h.val_acc, color="tab:green", label="val acc")
    ax2.set_ylabel("validation", color="tab:blue")
    ax2.set_ylim(0, 1)
    ax1.set_title("Training curve")
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.15))
    fig.tight_layout()
    fig.savefig(REPORTS / "training_curve.png", dpi=140)
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    y, p = predict_test_split()

    acc = float((y == p).mean())
    macro_f1 = float(f1_score(y, p, average="macro"))
    report_txt = classification_report(y, p, target_names=CLASSES, digits=3, zero_division=0)

    (REPORTS / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    (REPORTS / "metrics.json").write_text(
        json.dumps({"test_accuracy": round(acc, 4), "test_macro_f1": round(macro_f1, 4),
                    "n_test_clips": int(len(y))}, indent=2),
        encoding="utf-8",
    )
    plot_confusion(y, p)
    plot_training_curve()

    print(report_txt)
    print(f"test accuracy {acc:.3f}  macro-F1 {macro_f1:.3f}  (n={len(y)})")
    print(f"wrote figures + metrics to {REPORTS}")


if __name__ == "__main__":
    main()
