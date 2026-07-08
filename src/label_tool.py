"""Keyboard-driven clip labeling tool.

Plays each unlabeled clip on loop; press a key to label it and move on.
Progress is saved to labels_manual.csv after every clip — quit and resume anytime.

Usage:
    python label_tool.py            # label V2-V5 clips
    python label_tool.py V3         # label only one folder

Keys:
    1 normal   2 copying   3 gesture   4 mobile   5 notes
    6 other/unclear        space pause/resume     u undo last
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
FOLDERS = ["V2", "V3", "V4", "V5"]


def collect(folders):
    clips = []
    for v in folders:
        d = ARCHIVE / v / v
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".mp4", ".mov"):
                clips.append((v, f))
    return clips


def main():
    folders = [a for a in sys.argv[1:] if a in FOLDERS] or FOLDERS
    done = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame(columns=["folder", "path", "label"])
    done_paths = set(done["path"])
    clips = [(v, f) for v, f in collect(folders) if str(f) not in done_paths]
    print(f"{len(clips)} clips to label ({len(done_paths)} already done)")
    if not clips:
        return

    hud = "1=normal 2=copy 3=gesture 4=mobile 5=notes 6=other | s=skip u=undo q=quit"
    i = 0
    while i < len(clips):
        v, f = clips[i]
        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))
        label, paused, frame = None, False, None
        while label is None:
            if not paused:
                ok, frame = cap.read()
                if not ok:  # loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            disp = frame.copy()
            h, w = disp.shape[:2]
            if max(h, w) > 1000:
                s = 1000 / max(h, w)
                disp = cv2.resize(disp, (int(w * s), int(h * s)))
            cv2.putText(disp, f"[{i + 1}/{len(clips)}] {v}/{f.name}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, hud, (10, disp.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow("labeler", disp)
            k = cv2.waitKey(delay) & 0xFF
            if k in KEYMAP:
                label = KEYMAP[k]
            elif k == ord("s"):
                label = "__skip__"
            elif k == ord(" "):
                paused = not paused
            elif k == ord("u") and len(done):
                done = done.iloc[:-1]
                done.to_csv(OUT_CSV, index=False)
                print("undid last label")
            elif k == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print(f"saved {len(done)} labels to {OUT_CSV}")
                return
        cap.release()
        if label != "__skip__":
            done = pd.concat([done, pd.DataFrame([{"folder": v, "path": str(f), "label": label}])],
                             ignore_index=True)
            done.to_csv(OUT_CSV, index=False)
        i += 1
    cv2.destroyAllWindows()
    print(f"all done - {len(done)} labels in {OUT_CSV}")


if __name__ == "__main__":
    main()
