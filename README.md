# Pneumonia X-ray – Deep Learning (Keras/TensorFlow)

Convolutional Neural Network to detect Pneumonia from chest X-ray images.  

> Includes training/eval scripts, a ready-to-run MobileNetV2 baseline, plots, and saved models.

---

## Dataset
Use the **Chest X-Ray Images (Pneumonia)** dataset (Kermany et al.) from Kaggle or Mendeley.  
Expected folder structure after download/extract:

```
data/chest_xray/
├── train/
│   ├── NORMAL/ *.jpeg
│   └── PNEUMONIA/ *.jpeg
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Train
python src/train.py --data_dir data/chest_xray --epochs 20 --img_size 224 --batch 32 --model_out models/mobilenetv2.keras

# 3) Evaluate on test set
python src/evaluate.py --data_dir data/chest_xray --model_path models/mobilenetv2.keras

# 4) Predict a single image
python src/predict.py --model_path models/mobilenetv2.keras --image_path path/to/xray.jpg
```

---

## Results (example run)
Training history from a previous experiment is included (`history.json`)

> *The performance metrics obtained are particularly noteworthy: precision (~92%), recall (~88%), and F1-score (~90%) indicate robust classification capabilities. The implementation of overfitting mitigation strategies, including L2 regularization and dynamic learning rate adjustment through ReduceLROnPlateau, proved effective in maintaining model stability while preserving predictive power. The early stopping mechanism ensured optimal model selection while preventing performance degradation.
*

---

## Model
- **Architecture:** CNN with Con2D, MaxPooling2D, Dropout, GlobalMaxPooling2D  
- **Head:** GlobalAveragePooling2D + Dense w/ Dropout  
- **Loss:** Binary cross-entropy  
- **Metrics:** Accuracy, Loss, Precision, Recall, F1-Score, Confusion Matrix, Calibration Plot, ROC Curve
- **Augmentations:** Performed using ImageDataGenerator  
- **Class imbalance:** Used Oversampling  

---

## Repo Structure
```
src/                # training, eval, predict scripts
models/             # saved models (sample artifacts included)
notebooks/          # EDA/experiments (user-provided)
data/               # put dataset here (gitignored)
plots/              # generated charts
history.json        # example training log (user-provided)
```

---

## Reproducibility
- Set seeds and determinism flags when needed (see `train.py --seed`).  
- Log history to `history.json` and export plots to `plots/`.  
- Use the included eval script to generate a confusion matrix & classification report.  

---

## Ethics & Usage
This repo is for **research/education** only.  

---
