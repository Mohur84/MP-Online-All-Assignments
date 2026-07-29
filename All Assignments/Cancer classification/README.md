# 🧠 Brain Tumor MRI Classification using CNN
### TensorFlow / Keras | 4-Class Multiclass Classification

---

## 📌 Project Overview

This project classifies brain MRI scans into **4 categories** using a
Convolutional Neural Network (CNN) with Transfer Learning:

| Class | Description |
|---|---|
| **Glioma** | Tumors in glial cells (most common brain tumor) |
| **Meningioma** | Tumors in the membranes around the brain |
| **Pituitary** | Tumors in the pituitary gland |
| **No Tumor** | Healthy brain scan |

---

## 📥 Step 1: Download the Dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click **Download** (you'll need a free Kaggle account)
3. Unzip the file — you'll get a folder with `Training/` and `Testing/`
4. Place it next to your Python file like this:

```
your_project/
├── brain_tumor_classification.py   ← the main code
├── requirements.txt
├── dataset/
│   ├── Training/
│   │   ├── glioma/          (~1300 images)
│   │   ├── meningioma/      (~1300 images)
│   │   ├── notumor/         (~1595 images)
│   │   └── pituitary/       (~1457 images)
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
```

---

## ⚙️ Step 2: Set Up Your Environment

### Option A — Local Machine (with GPU recommended)
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B — Google Colab (FREE GPU, recommended for beginners!)
1. Go to https://colab.research.google.com
2. Upload your dataset to Google Drive
3. Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. Update paths in the code:
```python
TRAIN_DIR = "/content/drive/MyDrive/dataset/Training"
TEST_DIR  = "/content/drive/MyDrive/dataset/Testing"
```
5. Enable GPU: Runtime → Change runtime type → T4 GPU

---

## ▶️ Step 3: Run the Project

```bash
python brain_tumor_classification.py
```

---

## 🏗️ How the Model Works (Beginner Explanation)

```
MRI Image (224×224 px)
        │
        ▼
 ┌──────────────────────────────┐
 │   EfficientNetB0 (Backbone) │  ← Pre-trained on 1M+ images
 │   Extracts visual features  │    (edges, shapes, textures)
 └──────────────────────────────┘
        │
        ▼
 Global Average Pooling          ← Compress features to a vector
        │
        ▼
 Dense(256) + Dropout            ← Learn tumor-specific patterns
        │
        ▼
 Dense(4) + Softmax              ← Output 4 probabilities
        │
        ▼
 [Glioma: 85%] [Meningioma: 5%] [NoTumor: 3%] [Pituitary: 7%]
```

### Why Transfer Learning?
- Training a CNN from scratch needs **millions** of images
- We only have ~5000 MRI images
- EfficientNetB0 already knows how to detect edges, shapes, and textures
- We just teach it the **final step**: distinguishing tumor types

---

## 📊 Output Files Generated

| File | Description |
|---|---|
| `sample_images.png` | Grid of sample MRI images from training set |
| `training_curves.png` | Accuracy & Loss graphs over epochs |
| `confusion_matrix.png` | How well model predicts each class |
| `best_brain_tumor_model.keras` | Saved best model checkpoint |
| `brain_tumor_classifier_final.keras` | Final trained model |

---

## 🔮 Predicting a New MRI Scan

After training, use this in the Python file:

```python
predict_single_image("path/to/your/scan.jpg", model, CLASS_NAMES)
```

Output example:
```
Prediction: glioma
Confidence: 91.3%

Prediction Breakdown:
  glioma      : ██████████████████████████    91.3%
  meningioma  : ██                             4.1%
  notumor     : █                              2.8%
  pituitary   : █                              1.8%
```

---

## 📈 Expected Results

With this architecture and dataset you should achieve:

| Metric | Expected Range |
|---|---|
| Training Accuracy | 95–99% |
| Validation Accuracy | 92–97% |
| Test Accuracy | 90–96% |

---

## 🗂️ Project Structure

```
brain_tumor_cnn_project/
├── brain_tumor_classification.py   ← Main training script
├── requirements.txt                ← Dependencies
├── README.md                       ← This file
├── dataset/                        ← Download from Kaggle
├── best_brain_tumor_model.keras    ← Generated after training
├── brain_tumor_classifier_final.keras
├── sample_images.png
├── training_curves.png
└── confusion_matrix.png
```

---

## 💡 Tips for Better Results

1. **Use GPU** — Training on CPU will take hours; Colab's free T4 GPU does it in ~20 min
2. **Don't change BATCH_SIZE to >32** unless you have >8GB GPU RAM
3. **EarlyStopping** will stop automatically if the model stops improving
4. **Class imbalance** — If one class has far more images, consider class weights

---

*Dataset credit: Masoud Nickparvar — Brain Tumor MRI Dataset (Kaggle)*
