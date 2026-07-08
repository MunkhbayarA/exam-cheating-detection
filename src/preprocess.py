"""Build labeled manifest from filename prefixes and cache fixed-size frame stacks.

Classes come from clip filename prefixes:
  nor -> normal, c -> copying, g -> gesture, m -> mobile, n -> notes
('s' clips and unrecognized names are skipped for phase 1.)

Each clip is decoded once: 24 frames sampled uniformly, center square crop,
resized to 144x144, saved as uint8 .npy. Training then never touches video files.
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
CACHE = ROOT / "data_cache"
N_FRAMES = 24
SIZE = 144

CLASS_MAP = {"nor": "normal", "c": "copying", "g": "gesture", "m": "mobile", "n": "notes"}
NAME_RE = re.compile(r"^(nor|c|g|m|n|s)(\d+)$", re.IGNORECASE)

SOURCES = [
    ("V1_raw", ARCHIVE / "V1" / "V1"),
    ("sample_224", ARCHIVE / "sample_of_our_preprocessed_data_224x224_cropped_clips" / "224x224_cropped_videos"),
]


def collect_clips():
    rows = []
    for source, folder in SOURCES:
        for f in sorted(folder.glob("*.mp4")):
            m = NAME_RE.match(f.stem)
            if not m:
                print(f"  skip (unrecognized name): {source}/{f.name}")
                continue
            prefix = m.group(1).lower()
            if prefix == "s":
                continue  # classroom-scene clips, unknown taxonomy for now
            rows.append({
                "path": str(f),
                "source": source,
                "clip_id": f"{prefix}{m.group(2)}",  # shared between raw & cropped versions
                "label": CLASS_MAP[prefix],
            })
    return pd.DataFrame(rows)


def extract_frames(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, N_FRAMES).astype(int)
    frames, want = [], set(idxs.tolist())
    pos = 0
    while len(frames) < N_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if pos in want:
            h, w = frame.shape[:2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            crop = frame[y0:y0 + side, x0:x0 + side]
            crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # duplicated indices (short clips) need multiple appends
            for _ in range(int((idxs == pos).sum()) - 1):
                frames.append(frames[-1])
        pos += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < N_FRAMES:  # pad short reads with last frame
        frames.append(frames[-1])
    return np.stack(frames[:N_FRAMES]).astype(np.uint8)


def main():
    CACHE.mkdir(exist_ok=True)
    df = collect_clips()
    print(f"{len(df)} labeled clips found")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))

    cache_paths, keep = [], []
    for i, row in df.iterrows():
        out = CACHE / f"{row.source}_{row.clip_id}.npy"
        if not out.exists():
            arr = extract_frames(row.path)
            if arr is None:
                print(f"  DECODE FAILED: {row.path}")
                keep.append(False)
                cache_paths.append("")
                continue
            np.save(out, arr)
        keep.append(True)
        cache_paths.append(str(out))
        if (i + 1) % 25 == 0:
            print(f"  cached {i + 1}/{len(df)}")

    df["cache"] = cache_paths
    df = df[np.array(keep)].reset_index(drop=True)
    df.to_csv(ROOT / "manifest.csv", index=False)
    print(f"wrote manifest.csv with {len(df)} clips -> cache in {CACHE}")


if __name__ == "__main__":
    sys.exit(main())
