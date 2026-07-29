# 🐾 Face Recognition with CNN
### Using LFW (Human Faces) + Wildlife Face Datasets

---

## 📌 What This Project Does

This project builds a **Convolutional Neural Network (CNN)** that can:
1. **Recognize human faces** using the LFW dataset (13,000+ photos)
2. **Recognize animal faces** using the Wildlife dataset
3. **Verify** whether two photos are of the same person/animal

---

## 🧠 How a CNN Recognizes Faces (Simple Explanation)

```
Your image → [Conv Layer 1] → Detects edges & lines
           → [Conv Layer 2] → Detects eyes, noses, ears
           → [Conv Layer 3] → Detects face structure
           → [Conv Layer 4] → Detects identity patterns
           → [FC Layers]    → "This is George Bush with 94% confidence"
```

Think of it like this:
- A baby first learns to see edges → then shapes → then faces → then "that's mom"
- A CNN does the exact same thing, but with math!

---

## 📁 Project Structure

```
face_recognition_project/
├── models/
│   └── cnn_model.py          ← The CNN architecture (the "brain")
├── utils/
│   └── data_loader.py        ← How we load and prepare images
├── train.py                  ← Run this to TRAIN the model
├── predict.py                ← Run this to USE the trained model
├── requirements.txt          ← Python packages to install
└── README.md                 ← You are here!
```

---

## 🚀 Getting Started (Step by Step)

### Step 1: Set Up Python Environment

```bash
# Create a virtual environment (keeps your project isolated)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train on LFW (Human Faces)

```bash
python train.py
```

This will:
- Auto-download the LFW dataset (~200 MB)
- Train for 30 epochs (~20 minutes on CPU, ~3 minutes on GPU)
- Save the best model to `checkpoints/best_model.pth`
- Print accuracy after each epoch

**Expected output:**
```
📥 Loading LFW dataset...
✅ LFW loaded: 3023 images, 62 people
🧠 Model: 2,847,946 parameters

Epoch 1/30 | LR: 0.001000
  ✅ Train → Loss: 2.1834  Acc: 42.3%
     Val   → Loss: 1.9201  Acc: 51.2%  (45.2s)

Epoch 10/30 | LR: 0.001000
  ✅ Train → Loss: 0.4123  Acc: 88.1%
     Val   → Loss: 0.5891  Acc: 84.3%
...
```

### Step 4: Make Predictions

```bash
# Recognize a face in an image
python predict.py --image path/to/face.jpg

# Verify if two images are the same person
python predict.py --verify photo1.jpg photo2.jpg

# Process a whole folder
python predict.py --batch my_images/
```

---

## 🐆 Using the Wildlife Dataset

If you have wildlife face images, organize them like this:

```
wildlife_data/
├── lion/          ← folder name = species label
│   ├── lion1.jpg
│   └── lion2.jpg
├── tiger/
│   └── tiger1.jpg
└── elephant/
    └── ele1.jpg
```

Then train on it:
```python
# In train.py, change:
CONFIG["dataset"] = "wildlife"
```

---

## 📊 Understanding Your Results

| Metric | What It Means | Good Value |
|--------|--------------|------------|
| Loss | How wrong the model is | Lower is better |
| Train Acc | Accuracy on training data | Should be high (>90%) |
| Val Acc | Accuracy on unseen data | Key metric (>75% is good) |
| Gap | Train Acc - Val Acc | If >20%, model is overfitting |

**Overfitting** = model memorized training data but fails on new images.
Fix: add more data augmentation, increase dropout, use fewer epochs.

---

## 🔧 Tips & Troubleshooting

| Problem | Solution |
|---------|----------|
| In case of low accuracy (<90%) | Increase epochs or lower learning rate |
| Training too slow | Enable GPU or reduce batch size |
| "CUDA out of memory" | Reduce batch_size to 16 or 8 |
| Overfitting (high train, low val) | Increase dropout or add more augmentation |
| Dataset not downloading | Check internet connection; try VPN |

---

## 📚 Learning Resources

- **What is a CNN?** → https://cs231n.github.io/convolutional-networks/
- **LFW Dataset info** → http://vis-www.cs.umass.edu/lfw/
- **PyTorch tutorials** → https://pytorch.org/tutorials/
- **Face recognition paper** → DeepFace (Facebook, 2014)

---

*Built with PyTorch • LFW Dataset • Wildlife Faces*
