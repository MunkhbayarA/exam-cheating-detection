# Model Card — Exam Behavior Classifier

## Overview
- **Task:** single-student exam-behavior video classification (5 classes).
- **Classes:** `normal`, `copying`, `gesture`, `mobile`, `notes`.
- **Architecture:** R(2+1)D-18, fine-tuned from Kinetics-400 pretrained weights
  (torchvision `r2plus1d_18`). The final FC layer is replaced with a 5-way head.
- **Input:** 16 frames sampled uniformly from a clip, center-cropped to 112×112,
  normalized with Kinetics statistics.

## Training Data
- Source: Kaggle **ExamCheating_MultiV** dataset (V1 + a 224×224 cropped-clip sample)
  plus manually-labeled V2 clips.
- Labels for V1/sample come from filename prefixes; V2 is hand-labeled with `label_tool.py`.
- **Split:** group-aware, stratified by class. Raw and cropped views of the same clip share
  a `clip_id` and are always kept in the same split, preventing train/test leakage.

## Evaluation
- Held-out test split, reported as accuracy + macro-F1 + per-class report + confusion matrix.
- **Baseline (2026-07-08, 221 clips, before V2 labeling):** test accuracy **81%**, macro-F1 **0.79**.
- _Post-V2 numbers will replace these once the retrain runs._

## Intended Use
- Research and education: exploring video-based action recognition on a small, noisy dataset.
- A teaching example of a reproducible fine-tuning pipeline with leakage-safe splits.

## Out-of-Scope / Prohibited Use
- **Not** an automated proctoring or disciplinary tool. Outputs must not be used as evidence
  to accuse or penalize a student.
- Not validated for deployment, real-time use, or any high-stakes decision.

## Limitations & Risks
- **Small dataset** → high variance, limited generalization to new rooms/cameras/students.
- **Potential bias**: behavior appearance varies by culture, body type, and disability;
  the training set does not control for this.
- **Ambiguity**: "gesture" vs "copying" vs "normal" are subjective and can be mislabeled.
- **Privacy**: behavior surveillance raises consent and privacy concerns beyond model accuracy.

## Ethical Guidance
Treat any prediction as, at most, a suggestion for a human to review — never an automated
judgment. See the repo's *Limitations & Responsible Use* section.
