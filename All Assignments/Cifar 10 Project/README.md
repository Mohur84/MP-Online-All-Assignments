# 🚀 CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras" />
  <img src="https://img.shields.io/badge/CNN-Computer%20Vision-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

# 📖 Project Overview

This project presents a complete implementation of a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10** dataset. The model is developed entirely using **TensorFlow** and **Keras**, demonstrating the complete deep learning workflow—from preprocessing the dataset to training, evaluating, and saving the final model.

The CNN learns meaningful image features automatically through multiple convolutional layers and predicts one of the ten object categories with high accuracy.

---

# 🎯 Objectives

- Build a CNN from scratch using TensorFlow/Keras.
- Perform image preprocessing and normalization.
- Improve generalization using Batch Normalization and Dropout.
- Reduce overfitting using Data Augmentation.
- Train and evaluate the model efficiently.
- Visualize learning curves and classification performance.
- Save the trained model for future inference.

---

# 🖼 Dataset

**Dataset:** CIFAR-10

Official Website:

https://www.cs.toronto.edu/~kriz/cifar.html

### Dataset Information

| Property | Value |
|-----------|-------|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 × 32 Pixels |
| Color Channels | RGB |
| Number of Classes | 10 |

---

# 📦 CIFAR-10 Classes

| ID | Class |
|----|-------|
| 0 | ✈ Airplane |
| 1 | 🚗 Automobile |
| 2 | 🐦 Bird |
| 3 | 🐱 Cat |
| 4 | 🦌 Deer |
| 5 | 🐶 Dog |
| 6 | 🐸 Frog |
| 7 | 🐴 Horse |
| 8 | 🚢 Ship |
| 9 | 🚚 Truck |

---

# 🧠 CNN Architecture

```text
Input (32×32×3)
│
├── Conv2D (32 Filters, 3×3, ReLU)
├── Batch Normalization
├── Conv2D (32 Filters)
├── MaxPooling2D
├── Dropout
│
├── Conv2D (64 Filters)
├── Batch Normalization
├── Conv2D (64 Filters)
├── MaxPooling2D
├── Dropout
│
├── Conv2D (128 Filters)
├── Batch Normalization
├── MaxPooling2D
├── Dropout
│
├── Flatten
├── Dense (256)
├── Dropout
└── Dense (10)
    ↓
 Softmax
```

---

# ⚙ Features

- TensorFlow 2.x & Keras Sequential API
- Custom CNN Architecture
- Batch Normalization
- Dropout Regularization
- Image Normalization
- Data Augmentation
- Early Stopping
- ReduceLROnPlateau
- Model Checkpointing
- Accuracy & Loss Visualization
- Confusion Matrix
- Classification Report
- Saved Trained Model

---

# 📊 Model Performance

| Metric | Result |
|----------|---------|
| Framework | TensorFlow 2.21 |
| Model | Custom CNN |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Test Accuracy | **≈80%** |

> Results may vary depending on hardware and random initialization.

---

# 📂 Project Structure

```text
CIFAR10-Image-Classification-CNN/
│
├── assets/
├── notebook/
│   └── Mohur_Datta_CIFAR10_CNN.ipynb
├── src/
│   └── cifar10_cnn.py
├── model/
│   └── cifar10_final_model.keras
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/CIFAR10-Image-Classification-CNN.git
cd CIFAR10-Image-Classification-CNN
pip install -r requirements.txt
```

---

# ▶ Running the Project

```bash
python src/cifar10_cnn.py
```

or open:

```text
notebook/Mohur_Datta_CIFAR10_CNN.ipynb
```

---

# 💾 Loading the Saved Model

```python
import tensorflow as tf

model = tf.keras.models.load_model("model/cifar10_final_model.keras")
```

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 🔮 Future Improvements

- ResNet50
- EfficientNet
- MobileNetV2
- Streamlit
- Flask REST API
- Docker
- ONNX Export
- TensorBoard

---

# 👨‍💻 Author

**Mohur Datta**

**B.Tech – Computer Science Engineering (Artificial Intelligence & Machine Learning)**

**VIT Bhopal University**

**Registration Number:** **23BAI11091**

---

# 📜 License

Licensed under the **MIT License**.

---

# ⭐ Support

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub.

<p align="center">
<b>Made with ❤️ using TensorFlow & Keras</b>
</p>
