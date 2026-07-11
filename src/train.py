"""Fine-tune Kinetics-pretrained R(2+1)D-18 on the 5-class cheating clips."""
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from dataset import CLASSES, ClipDataset, make_splits

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 42
BATCH = 8
EPOCHS = 40
PATIENCE = 8
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast("cuda"):
        for x, y in loader:
            out = model(x.to(device, non_blocking=True))
            ps.append(out.argmax(1).cpu())
            ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return f1_score(y, p, average="macro"), (y == p).mean()


def main():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available"

    df = make_splits(ROOT / "manifest.csv", seed=SEED)
    df.to_csv(ROOT / "manifest_splits.csv", index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    tr = ClipDataset(df[df.split == "train"], train=True)
    va = ClipDataset(df[df.split == "val"], train=False)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, persistent_workers=True, drop_last=True)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2,
                    pin_memory=True, persistent_workers=True)

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)

    counts = df[df.split == "train"].label.value_counts().reindex(CLASSES).values
    weights = torch.tensor((counts.sum() / (len(CLASSES) * counts)), dtype=torch.float32, device=device)
    print("class weights:", dict(zip(CLASSES, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler()

    best_f1, best_epoch = 0.0, -1
    history = []  # per-epoch metrics -> reports/history.csv for the training-curve plot
    for epoch in range(EPOCHS):
        model.train()
        t0, losses = time.time(), []
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        val_f1, val_acc = run_eval(model, vl, device)
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_f1": val_f1, "val_acc": val_acc})
        marker = ""
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"model": model.state_dict(), "classes": CLASSES, "epoch": epoch,
                        "val_f1": val_f1}, ROOT / "best_model.pt")
            marker = "  <- saved"
        print(f"epoch {epoch:02d}  loss {train_loss:.3f}  val_f1 {val_f1:.3f}  "
              f"val_acc {val_acc:.3f}  ({time.time() - t0:.0f}s){marker}", flush=True)

        if epoch - best_epoch >= PATIENCE:
            print(f"early stop: no val improvement for {PATIENCE} epochs")
            break

    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(REPORTS / "history.csv", index=False)
    print(f"best val macro-F1 {best_f1:.3f} at epoch {best_epoch}")
    print(f"wrote training history -> {REPORTS / 'history.csv'}")


if __name__ == "__main__":
    main()
