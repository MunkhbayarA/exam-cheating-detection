"""CPU-only unit tests for the data pipeline. No dataset or GPU required."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------- split integrity
def _synthetic_manifest(tmp_path):
    """A manifest with 3 classes, 8 clips each, where every clip has a raw AND a
    cropped row sharing the same clip_id (the real dataset's structure)."""
    rows = []
    for label in ("normal", "copying", "mobile"):
        for k in range(8):
            cid = f"{label}{k}"
            for source in ("V1_raw", "sample_224"):
                rows.append({"path": f"{source}/{cid}.mp4", "source": source,
                             "clip_id": cid, "label": label,
                             "cache": f"{source}_{cid}.npy"})
    csv = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_split_has_no_clip_id_leakage(tmp_path):
    """The core invariant: a clip_id (and thus its raw+cropped views) must live in
    exactly one split, or test metrics are inflated by train/test leakage."""
    from dataset import make_splits
    df = make_splits(_synthetic_manifest(tmp_path), seed=0)
    splits_per_clip = df.groupby("clip_id")["split"].nunique()
    assert (splits_per_clip == 1).all(), "a clip_id leaked across splits"


def test_split_assigns_every_clip(tmp_path):
    from dataset import make_splits
    df = make_splits(_synthetic_manifest(tmp_path), seed=0)
    assert df["split"].isin({"train", "val", "test"}).all()
    assert set(df["split"]) == {"train", "val", "test"}


def test_split_is_deterministic(tmp_path):
    from dataset import make_splits
    csv = _synthetic_manifest(tmp_path)
    a = make_splits(csv, seed=42).sort_values("clip_id")["split"].tolist()
    b = make_splits(csv, seed=42).sort_values("clip_id")["split"].tolist()
    assert a == b


# ---------------------------------------------------------------- tensor shape
def test_frames_to_tensor_shape_and_range():
    from dataset import CROP, N_FRAMES_OUT
    from inference import frames_to_tensor
    frames = [np.random.randint(0, 256, (CROP, CROP, 3), dtype=np.uint8)
              for _ in range(N_FRAMES_OUT)]
    t = frames_to_tensor(frames)
    assert tuple(t.shape) == (3, N_FRAMES_OUT, CROP, CROP)
    assert t.dtype.is_floating_point


# ---------------------------------------------------------------- manual ingest
def test_collect_manual_filters_and_dedups(tmp_path, monkeypatch):
    import preprocess
    csv = tmp_path / "labels_manual.csv"
    pd.DataFrame([
        {"folder": "V2", "path": "archive/V2/V2/1.mp4", "label": "copying"},
        {"folder": "V2", "path": "archive/V2/V2/2.mp4", "label": "mobile"},
        {"folder": "V2", "path": "archive/V2/V2/3.mp4", "label": "other"},   # dropped
        {"folder": "V2", "path": "archive/V2/V2/1.mp4", "label": "notes"},   # dup -> last
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(preprocess, "MANUAL_CSV", csv)
    man = preprocess.collect_manual()
    assert len(man) == 2                                   # other dropped, dup collapsed
    assert set(man["label"]) == {"mobile", "notes"}        # V2_1 kept 'notes' (last)
    assert set(man["clip_id"]) == {"V2_1", "V2_2"}
    assert (man["source"] == "manual").all()


def test_collect_manual_missing_file(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "MANUAL_CSV", tmp_path / "does_not_exist.csv")
    assert preprocess.collect_manual().empty


# ---------------------------------------------------------------- frame sampling
def test_sample_frames_shape(tmp_path):
    """Decode a synthetic non-square video and confirm sample_frames returns the
    right number of square, correctly-sized frames. Skips if no video codec."""
    import cv2

    from dataset import CROP, N_FRAMES_OUT
    from inference import sample_frames

    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (200, 150))
    if not writer.isOpened():
        pytest.skip("no mp4 codec available in this environment")
    for i in range(30):
        frame = np.full((150, 200, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = sample_frames(path)
    assert len(frames) == N_FRAMES_OUT
    assert all(f.shape == (CROP, CROP, 3) for f in frames)
