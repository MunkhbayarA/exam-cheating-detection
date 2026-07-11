---
title: Exam Behavior Classifier
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🎓 Exam Behavior Classifier

Upload a short clip of a single student and an R(2+1)D-18 video CNN (fine-tuned from
Kinetics-400) predicts one of five exam behaviors: **normal, copying, gesture, mobile, notes**.

⚠️ **Research & education demo — not a proctoring system.** Trained on a small dataset;
treat any output as a suggestion to review, never as evidence of cheating.

Source, training pipeline, and model card:
[github.com/MunkhbayarA/exam-cheating-detection](https://github.com/MunkhbayarA/exam-cheating-detection)

<!--
Deploy checklist (done from the project root once an HF account + token exist):
  1. pip install huggingface_hub
  2. hf auth login            (paste the write token)
  3. Create the Space:  hf repo create exam-behavior-classifier --repo-type space --space-sdk gradio
  4. Upload files:
       app/app.py            -> app.py       (adjust the src/ import path or copy src modules)
       app/requirements.txt  -> requirements.txt
       app/SPACE_README.md   -> README.md
       src/dataset.py, src/inference.py -> src/
       best_model.pt         -> best_model.pt   (via LFS, ~125MB)
  5. The free CPU tier is enough; first build takes a few minutes.
-->
