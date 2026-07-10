"""Keyboard-driven clip labeling tool.

Plays each unlabeled clip on loop; press a key to label it and move on.
Progress is saved to labels_manual.csv after every clip — quit and resume anytime.

Usage:
    python label_tool.py            # label V2 (the label-ready mixed-bag clips)
    python label_tool.py V3         # label a specific folder (V3-V5 are wide-angle
                                    #   lecture halls — see README, not recommended yet)

Keys:
    1 normal   2 copying   3 gesture   4 mobile   5 notes
    6 other/unclear        space pause/resume     u undo (step back one clip)
    s skip (decide later)  q quit
"""
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
OUT_CSV = ROOT / "labels_manual.csv"

KEYMAP = {
    ord("1"): "normal", ord("2"): "copying", ord("3"): "gesture",
    ord("4"): "mobile", ord("5"): "notes", ord("6"): "other",
}
VALID_FOLDERS = ["V2", "V3", "V4", "V5"]
DEFAULT_FOLDERS = ["V2"]  # V3-V5 are wide-angle halls; clip-level labels not valid yet
HUD = "1=normal 2=copy 3=gesture 4=mobile 5=notes 6=other | s=skip u=undo q=quit"


def collect(folders):
    clips = []
    for v in folders:
        d = ARCHIVE / v / v
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".mp4", ".mov"):
                clips.append((v, f))
    return clips


def label_one(v, f, i, total):
    """Show one clip on loop until the user presses a key.

    Returns one of: ("label", name) | ("skip",) | ("undo",) | ("quit",).
    """
    cap = cv2.VideoCapture(str(f))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    paused, frame = False, None
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:  # loop back to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            disp = frame.copy()
            h, w = disp.shape[:2]
            if max(h, w) > 1000:
                s = 1000 / max(h, w)
                disp = cv2.resize(disp, (int(w * s), int(h * s)))
            cv2.putText(disp, f"[{i + 1}/{total}] {v}/{f.name}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, HUD, (10, disp.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("labeler", disp)
            k = cv2.waitKey(delay) & 0xFF
            if k in KEYMAP:
                return ("label", KEYMAP[k])
            if k == ord("s"):
                return ("skip",)
            if k == ord("u"):
                return ("undo",)
            if k == ord("q"):
                return ("quit",)
            if k == ord(" "):
                paused = not paused
    finally:
        cap.release()


def save(done):
    done.to_csv(OUT_CSV, index=False)


def main():
    folders = [a for a in sys.argv[1:] if a in VALID_FOLDERS] or DEFAULT_FOLDERS
    done = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame(columns=["folder", "path", "label"])
    done_paths = set(done["path"])
    clips = [(v, f) for v, f in collect(folders) if str(f) not in done_paths]
    print(f"labeling {folders}: {len(clips)} clips to go ({len(done_paths)} already done)")
    if not clips:
        print("nothing to label — all done.")
        return

    history = []  # stack of ("label", i) | ("skip", i) — enables real undo across clips
    i = 0
    while i < len(clips):
        v, f = clips[i]
        res = label_one(v, f, i, len(clips))
        action = res[0]

        if action == "quit":
            break
        if action == "undo":
            if not history:
                print("nothing to undo")
                continue
            kind, prev_i = history.pop()
            if kind == "label":  # remove the row we saved for that clip
                done = done.iloc[:-1].reset_index(drop=True)
                save(done)
            i = prev_i
            print(f"undo -> re-showing {clips[i][1].name}")
            continue
        if action == "skip":
            history.append(("skip", i))
            i += 1
            continue

        # action == "label"
        label = res[1]
        done = pd.concat(
            [done, pd.DataFrame([{"folder": v, "path": str(f), "label": label}])],
            ignore_index=True,
        )
        save(done)
        history.append(("label", i))
        i += 1

    cv2.destroyAllWindows()
    n_labeled = len(done) - len(done_paths)
    print(f"session done — {n_labeled} new labels this run, {len(done)} total in {OUT_CSV}")


if __name__ == "__main__":
    main()
