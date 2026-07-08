"""Dataset over cached frame stacks with group-aware stratified splits."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ["normal", "copying", "gesture", "mobile", "notes"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Kinetics-400 normalization used by torchvision video models
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

N_FRAMES_OUT = 16
CROP = 112


def make_splits(manifest_csv, seed=42, val_frac=0.15, test_frac=0.15):
    """Split at clip_id (group) level, stratified by label, so the raw and
    cropped versions of the same clip always land in the same split."""
    df = pd.read_csv(manifest_csv)
    rng = np.random.RandomState(seed)
    groups = df.drop_duplicates("clip_id")[["clip_id", "label"]]
    split_of = {}
    for label, g in groups.groupby("label"):
        ids = g["clip_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for cid in ids[:n_test]:
            split_of[cid] = "test"
        for cid in ids[n_test:n_test + n_val]:
            split_of[cid] = "val"
        for cid in ids[n_test + n_val:]:
            split_of[cid] = "train"
    df["split"] = df["clip_id"].map(split_of)
    return df


class ClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = np.load(row.cache)  # (24, 144, 144, 3) uint8
        n = arr.shape[0]

        if self.train:
            # temporal jitter: random contiguous-ish subsample of 16 frames
            start = np.random.randint(0, n - N_FRAMES_OUT + 1)
            idx = np.arange(start, start + N_FRAMES_OUT)
            y0 = np.random.randint(0, arr.shape[1] - CROP + 1)
            x0 = np.random.randint(0, arr.shape[2] - CROP + 1)
        else:
            idx = np.linspace(0, n - 1, N_FRAMES_OUT).astype(int)
            y0 = x0 = (arr.shape[1] - CROP) // 2

        clip = arr[idx, y0:y0 + CROP, x0:x0 + CROP]  # (16, 112, 112, 3)

        if self.train and np.random.rand() < 0.5:
            clip = clip[:, :, ::-1]  # horizontal flip (copying stays copying)

        t = torch.from_numpy(np.ascontiguousarray(clip)).float().div_(255)
        t = t.permute(3, 0, 1, 2)  # (C, T, H, W)
        if self.train:
            # mild brightness/contrast jitter
            t = (t * (0.8 + 0.4 * torch.rand(1)) + 0.1 * (torch.rand(1) - 0.5)).clamp_(0, 1)
        t = (t - MEAN) / STD
        return t, CLASS_TO_IDX[row.label]
