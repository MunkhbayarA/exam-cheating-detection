# Exam Cheating Detection

Video classification of exam-hall behavior using a fine-tuned R(2+1)D-18 network.
Classifies 3-4 second clips into five classes: **normal, copying, gesture, mobile, notes**.

## Dataset

[ExamCheating_MultiV](https://www.kaggle.com/datasets/rimmajeed/examcheating-multiv-video-based-dataset) (Kaggle, ~24GB — not included in this repo).
Download and extract it to `archive/` so the layout is:

```
archive/
  V1/V1/*.mp4            labeled clips (filename prefix = class)
  V2/V2/*                mixed clips
  V3..V5/                raw wide-angle lecture hall footage (unlabeled)
  sample_of_our_preprocessed_data_224x224_cropped_clips/
```

Labels are encoded in filename prefixes: `nor`=normal, `c`=copying, `g`=gesture,
`m`=mobile, `n`=notes.

## Results

Trained on 221 labeled clips (V1 + preprocessed sample), grouped split so raw and
cropped versions of the same clip never cross the train/test boundary.

| Metric | Value |
|---|---|
| Test accuracy | 81.2% |
| Macro F1 | 0.79 |
| Best class | gesture (F1 1.00) |
| Weakest class | normal (F1 0.55) |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install opencv-python pandas scikit-learn matplotlib tqdm
```

Requires an NVIDIA GPU (developed on an RTX 3080 10GB) and ffmpeg.

## Usage

```powershell
cd src
..\.venv\Scripts\python preprocess.py    # decode clips once into data_cache/
..\.venv\Scripts\python train.py         # fine-tune R(2+1)D-18, saves best_model.pt
..\.venv\Scripts\python evaluate.py      # test-set report + confusion matrix
..\.venv\Scripts\python predict.py <video.mp4>   # classify any clip
..\.venv\Scripts\python label_tool.py V2         # keyboard labeling tool for unlabeled clips
```

## Pipeline

1. `preprocess.py` — samples 24 frames per clip, center-crops, caches as uint8 arrays
2. `dataset.py` — grouped stratified splits, temporal jitter / crop / flip augmentation
3. `train.py` — Kinetics-400 pretrained R(2+1)D-18, class-weighted loss, AMP, early stopping
4. `evaluate.py` — per-class precision/recall/F1, confusion matrix, misclassified clip list
5. `label_tool.py` — plays unlabeled clips, press 1-5 to label, resumable CSV

## Roadmap

- Label V2 + V3-V5 footage (`label_tool.py`) and retrain
- Person-detection + crop stage (YOLO) so the classifier works on multi-student
  wide shots (V3-V5)
