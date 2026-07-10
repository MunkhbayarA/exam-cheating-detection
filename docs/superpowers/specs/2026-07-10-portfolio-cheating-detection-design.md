# Portfolio-Grade Cheating-Detection Repo — Design

**Date:** 2026-07-10
**Status:** Approved (design), pending spec review
**Author:** Munkhbayar (with Claude)

## Goal

Turn the existing exam-cheating video classifier into a **portfolio-grade GitHub repo** that
impresses AI/ML recruiters and senior engineers. Two threads:

1. **Substance** — finish the dataset by manually labeling V2, retrain, report honest metrics.
2. **Presentation** — live demo, polished README, reproducibility, tests/CI, and an honest
   ethics/limitations section.

The maintainer (Munkhbayar) does the labeling (human judgment task); Claude builds all tooling,
scaffolding, and runs the GPU retrain.

## Non-Goals

- No per-person cropping of V3/V4/V5 lecture-hall footage (523 clips). Those stay unlabeled;
  clip-level labels are semantically invalid for many-student wide-angle frames. Documented as
  future work.
- No new model architecture. The R(2+1)D-18 pipeline stays; we improve data + presentation.
- No inflated metrics. Results are reported truthfully against the 81% / macro-F1 0.79 baseline.

## Current State (verified 2026-07-10)

- Repo at `C:\Users\munkh\OneDrive\Documents\Cheating detection project\`, git repo, branch
  `feature/portfolio-upgrade` off `main`. GitHub: private `MunkhbayarA/exam-cheating-detection`.
- Pipeline in `src/`: `preprocess.py`, `dataset.py`, `train.py`, `evaluate.py`, `predict.py`,
  `label_tool.py`. Baseline `best_model.pt` (221 clips, test acc 81%, macro-F1 0.79).
- `.venv` with torch 2.5.1+cu121, OpenCV 5.0.0 (GUI-capable), pandas. RTX 3080 10GB.
- No `labels_manual.csv` yet. V2 = 30 clips named `1.mp4`…`30.mp4` (no label prefix → need eyes).
- Data flow: `preprocess.py` reads V1 + sample_224, labels from filename prefixes → caches
  fixed-size frame stacks as `.npy` → `manifest.csv`. `dataset.make_splits` does group-aware
  stratified split by `clip_id`. `train.py` fine-tunes, saves `best_model.pt`. `evaluate.py`
  reports test metrics.

## Architecture / Components

### A. Substance
- **`label_tool.py` fixes:** default to labeling **V2 only** (V3–V5 accessible via explicit arg but
  not default, to prevent labeling 553 unlabelable clips); fix `undo` so `u` steps back to the
  previous clip and re-shows it (currently it deletes the prior label but moves forward, so mistakes
  can't be corrected). Keep loop playback, per-clip autosave, resume, skip.
- **`preprocess.py` extension:** after the existing prefix-based sources, if `labels_manual.csv`
  exists, ingest each row → decode → cache `.npy` → append to `manifest.csv` with
  `source="manual_V2"`, `clip_id="V2_<stem>"` (no collision with V1 ids), `label` from CSV. Drop
  `other`/unrecognized rows; only the 5 trained classes (normal, copying, gesture, mobile, notes)
  flow in. Downstream (`dataset`, `train`, `evaluate`) untouched — they just see more rows.
- **Reporting artifacts** → `reports/`: training-curve plot, confusion matrix, per-class F1 table,
  Grad-CAM "what the model looks at" overlays on sample clips.

### B. Live Demo
- **`app/` Gradio app:** upload a clip → sampled-frame preview + predicted class + confidence bars.
  CPU inference (33M params, short clips). Runs locally with no account.
- **Deploy to HuggingFace Spaces** (Phase 4, needs maintainer's free HF account + token) → clickable
  "Try it live" badge in README. Fallback if no HF account: local run + recorded demo GIF.

### C. README (storefront)
- Hero: one-line pitch, demo GIF, live-demo badge, results table up top.
- Sections: problem · dataset · method + architecture diagram · results (curves + confusion matrix)
  · Grad-CAM visuals · one-command reproduce · **Limitations & Responsible Use** · tech stack.

### D. Engineering Hygiene
- `requirements.txt` lock, MIT `LICENSE`, `model_card.md`.
- `pytest` tests (all CPU, tiny synthetic arrays): frame-extraction output shape; **split has zero
  `clip_id` leakage across train/val/test**; dataset `__getitem__` tensor shape/normalization.
- **GitHub Actions CI**: lint + tests on push (no GPU needed).
- Light reorg: `src/ app/ reports/ tests/ docs/ .github/workflows/`. No model-logic changes.

## Responsible-Use Framing (deliberate centerpiece)

The README includes an honest **Limitations & Responsible Use** section: this is a
research/education project, not a deployable proctoring system; small dataset; potential
demographic/behavioral bias; privacy implications of behavior surveillance; the model outputs a
suggestion, never an accusation. Presenting this signals maturity to serious reviewers.

## Data Flow

```
V2 clips ──(you, label_tool.py)──> labels_manual.csv
                                         │
V1 + sample_224 ──(prefix labels)──┐     │
                                   ▼     ▼
                            preprocess.py ──> data_cache/*.npy + manifest.csv
                                   │
                     dataset.make_splits (group-aware, stratified)
                                   │
                       train.py ──> best_model.pt
                                   │
              evaluate.py + reports (curves, confusion, Grad-CAM)
                                   │
                     app/ Gradio ──> HuggingFace Space
```

## Testing Strategy

- Unit tests run on CPU in CI with synthetic `.npy` arrays — no dataset or GPU required.
- Key invariant test: no `clip_id` appears in more than one split (leakage guard).
- Retrain metrics compared honestly against the 81% baseline; reported as-is.

## Phasing

1. **Tools:** `label_tool.py` fix + `preprocess.py` ingest. → maintainer starts labeling V2.
2. **Scaffold in parallel while labeling:** Gradio app, README skeleton, tests, CI, LICENSE,
   model card.
3. **After labeling:** retrain, generate figures + Grad-CAM, fill real numbers.
4. **Deploy:** HF Space (needs HF username/token), final README polish.

## Open Items / Maintainer Inputs

- HF username (Phase 4 only; free account, not a blocker earlier).
- Confirmed OK with the "Limitations & Responsible Use" honesty framing. ✅ (approved)
```
