# AgroVision Pro — Deployment Guide

## Folder Structure

```
agrovision_pro/
├── app.py                  # Main Streamlit application (all 17 phases)
├── image_validation.py     # Phase 2: Blur / brightness / contrast / resolution checks
├── leaf_segmentation.py    # Phase 3 & 4: Leaf coverage + background removal
├── gradcam.py              # Phase 9: Grad-CAM explainable AI heatmaps
├── train_efficientnet.py   # Phase 10 & 11: EfficientNetB0 training pipeline
├── utils.py                # Phase 5, 7, 8, 16: Preprocessing, top-k, reports
├── requirements.txt        # All Python dependencies
│
├── model.h5                # ← Your trained Keras model (place here)
├── class_names.json        # ← List of class label strings (place here)
├── knowledge_base.json     # ← Disease knowledge base (place here, optional)
└── weather_cache.json      # ← Auto-created by app at runtime
```

---

## Required Files (you supply)

| File | Format | Description |
|---|---|---|
| `model.h5` | Keras SavedModel | Your disease classification model |
| `class_names.json` | `["Class A", "Class B", ...]` | Ordered list matching model output |
| `knowledge_base.json` | `{"Disease Name": {"Symptoms": "...", "Causes": "...", "Prevention": "...", "Cure": "...", "Impact": "...", "Best Practices": "..."}}` | Advisory content (optional) |

---

## Local Setup

```bash
# 1. Create virtualenv
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your model.h5 and class_names.json in the project root

# 4. Run
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Push all files to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → select your repo.
3. Set **Main file path** to `app.py`.
4. Add a `packages.txt` in the root if OpenCV system libs are needed:
   ```
   libgl1
   libglib2.0-0
   ```
5. Deploy — Streamlit Cloud auto-installs `requirements.txt`.

---

## HuggingFace Spaces

1. Create a new Space → **Streamlit** SDK.
2. Upload all files.
3. HuggingFace auto-reads `requirements.txt` and runs `streamlit run app.py`.
4. Add a `packages.txt` if needed (same as above).

---

## Training a New Model (EfficientNetB0)

Prepare dataset with structure:
```
dataset/
├── train/
│   ├── Tomato_Early_Blight/
│   ├── Potato_Late_Blight/
│   └── ...
└── val/
    ├── Tomato_Early_Blight/
    └── ...
```

Then run:
```bash
python train_efficientnet.py \
    --data_dir dataset \
    --model_out model.h5 \
    --classes_out class_names.json \
    --epochs_tl 20 \
    --epochs_ft 30 \
    --batch 32
```

TensorBoard logs are saved to `logs/` automatically.

---

## Confidence Threshold

Default: **0.85** — predictions below this show an uncertainty warning but do not block.
Edit `CONFIDENCE_THRESHOLD` in `app.py` line ~390 to adjust.

---

## Features Summary

| Phase | Feature | File |
|---|---|---|
| 1 | Code audit & bug fixes | app.py |
| 2 | Image quality validation (blur/brightness/contrast/resolution) | image_validation.py |
| 3 | Leaf coverage analysis (HSV + contours) | leaf_segmentation.py |
| 4 | Background removal (GrabCut) | leaf_segmentation.py |
| 5 | Advanced preprocessing (CLAHE, sharpening, color norm) | utils.py |
| 6 | Confidence rejection threshold | app.py |
| 7 | Top-5 predictions bar chart | utils.py + app.py |
| 8 | Disease severity estimation (LOW/MEDIUM/HIGH/CRITICAL) | utils.py + app.py |
| 9 | Grad-CAM explainable AI heatmap | gradcam.py |
| 10 | EfficientNetB0 architecture + transfer learning | train_efficientnet.py |
| 11 | Data augmentation (flip/rotate/zoom/brightness/contrast) | train_efficientnet.py |
| 12 | Inference caching, model warmup, lazy loading | app.py |
| 13 | Professional glassmorphism UI (5 tabs, 5 KPI cards) | app.py |
| 14 | Farmer recommendation engine (multilingual) | app.py + utils.py |
| 15 | Analytics dashboard (trend, history, top-5 chart) | app.py |
| 16 | TXT / CSV / JSON report exports | utils.py + app.py |
| 17 | All files production-ready, no placeholders | All files |
