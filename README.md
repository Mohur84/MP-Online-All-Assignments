# All Assignments — Portfolio Overview

**Author:** Mohur Datta · **Registration Number:** 23BAI11091
**Program:** B.Tech Computer Science and Engineering (AI/ML)

This repository is a collection of 9 independent machine learning, deep learning, and reinforcement learning projects, spanning classical ML, computer vision, generative AI/RAG, and RL. Each subfolder is a self-contained project with its own dependencies, scripts, and (where applicable) its own README with detailed instructions.

---

## 📁 Repository Structure

```
All Assignments/
├── Adult Census Income/                              # Classical ML — income classification
├── Cancer classification/                             # Deep Learning — brain tumor MRI classification
├── CartPole-agent/                                     # Reinforcement Learning — PPO on CartPole-v1
├── Cifar 10 Project/                                   # Deep Learning — CNN image classification
├── Lunar Landing using Proximal Policy Optimization main/  # Reinforcement Learning — PPO on LunarLander
├── Movie-Recommendation-System-main/                   # ML + Web App — content-based recommender
├── RAG-Chatbot-main/                                   # GenAI — Retrieval-Augmented Generation chatbot
├── car price prediction render project/                # ML + Web App — car price regression, deployed
└── face recognition project/face_recognition_project/  # Deep Learning — CNN face recognition
```

---

## 📊 Project Summaries

### 1. Adult Census Income
**Type:** Classical ML (classification) · **Format:** Jupyter Notebook (`Mohur_Datta_23BAI11091.ipynb`)

Predicts whether an individual's annual income exceeds $50K using the Kaggle "Adult Census Income" dataset. Covers the full pipeline: data understanding, cleaning (handling `?` as missing values, mode imputation), one-hot encoding, an 80/20 stratified train-test split, and feature scaling. Trains and compares five algorithms — Logistic Regression, Decision Tree, Random Forest, KNN, and SVM — with performance evaluation across all of them.

- **Dataset:** Loaded automatically via `kagglehub`
- **Key libraries:** pandas, scikit-learn

---

### 2. Cancer Classification (Brain Tumor MRI)
**Type:** Deep Learning (CNN, transfer learning) · **Format:** Python script (`brain_tumor_classification.py`)

Classifies brain MRI scans into 4 classes — **Glioma**, **Meningioma**, **Pituitary**, **No Tumor** — using a CNN built on TensorFlow/Keras with transfer learning.

- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, manual download required — must be placed in a local `dataset/Training` / `dataset/Testing` structure)
- **Key libraries:** tensorflow, scikit-learn, seaborn, Pillow
- **License:** MIT (includes `LICENSE` file)

---

### 3. CartPole-agent
**Type:** Reinforcement Learning (PPO) · **Format:** Python scripts

Trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3 to solve the Gymnasium `CartPole-v1` environment.

- **Scripts:** `train.py` (train + save), `evaluate.py` (100-episode deterministic evaluation), `test.py` (live visual rendering)
- **Artifacts:** `ppo_model.zip` (trained weights), `evaluation_report.txt` (results)
- **Key libraries:** stable-baselines3[extra], gymnasium, matplotlib
- **Setup note:** Use a clean Conda environment (Python 3.10); quote `"stable-baselines3[extra]"` on macOS/Zsh to avoid glob errors.

---

### 4. Cifar 10 Project
**Type:** Deep Learning (CNN) · **Format:** Python script + Jupyter Notebook

A complete CNN pipeline for the CIFAR-10 image classification dataset (10 object categories), built with TensorFlow/Keras. Includes Batch Normalization, Dropout, and Data Augmentation to reduce overfitting, plus learning-curve and confusion-matrix visualizations.

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Deliverables:** `cifar10_cnn.py`, `notebook/CIFAR10_CNN.ipynb`, `confusion_matrix.png`
- **Key libraries:** tensorflow, scikit-learn, seaborn, pandas
- **License:** MIT (includes `LICENSE` file)

---

### 5. Lunar Landing using Proximal Policy Optimization
**Type:** Reinforcement Learning (PPO, continuous control) · **Format:** Python scripts

An Actor-Critic PPO agent trained to autonomously land a spacecraft in the Box2D-based `LunarLander-v3` Gymnasium environment (8-D observation space, 4 discrete actions).

- **Scripts:** `train.py`, `evaluate.py`, `test.py`, `plot_training.py`, `record_video.py`
- **Artifacts:** `models/ppo_lunarlander.zip`, `logs/training_monitor.csv`, `videos/rl-video-episode-0.mp4` (recorded landing)
- **Key libraries:** gymnasium[box2d], stable-baselines3, torch, moviepy
- **Environment:** Developed on Windows 11 (x86)

---

### 6. Movie Recommendation System
**Type:** ML + Web Application · **Format:** Flask web app

A content-based movie recommender that suggests similar movies using **TF-IDF vectorization** and **cosine similarity** over movie genres, served through a Flask web interface.

- **Deliverables:** `app.py` (Flask server), `recommender.py` (recommendation logic), `movies.csv` (dataset), `templates/index.html`, `static/style.css`
- **Key libraries:** Flask, pandas, numpy, scikit-learn, gunicorn
- **Deployment:** Configured for Render (`procfile` included)

---

### 7. RAG-Chatbot
**Type:** Generative AI / Retrieval-Augmented Generation · **Format:** Streamlit web app

A local, production-style RAG chatbot that ingests and indexes dense PDF documents, then answers questions grounded in their content.

- **Pipeline:** LangChain for orchestration, **Google Gemini** for embeddings and text generation, **ChromaDB** for local persistent vector storage
- **Features:** Self-throttling ingestion with backoff (to respect free-tier rate limits), 1000-char chunking with 150-char overlap, top-`k=6` retrieval, stateful Streamlit chat UI
- **Deliverables:** `ingest.py` (document processing → vector store), `app.py` (chat UI), `data/sample.pdf`, `.chroma_db/` (generated vector store)
- **Key libraries:** langchain, langchain-google-genai, chromadb, tiktoken, pypdf, streamlit
- **Setup note:** Requires a `.env` file with a Google Gemini API key (not included)

---

### 8. Car Price Prediction (Render Deployment)
**Type:** ML + Web Application (deployed) · **Format:** Flask web app

An end-to-end ML web app that predicts a car's resale price from its specifications, using a **Random Forest Regressor**, served via Flask/Gunicorn and deployed live on Render.

- **Live app:** https://car-price-prediction-app-00au.onrender.com
- **Deliverables:** `app.py`, `car_price_model.pkl` (serialized trained model), `templates/index.html`, `static/style.css`, `Procfile`, `runtime.txt`
- **Key libraries:** Flask, scikit-learn, pandas, numpy, gunicorn, opendatasets
- **Stack:** Python 3.10, Gunicorn (WSGI), Render (cloud hosting)

---

### 9. Face Recognition Project
**Type:** Deep Learning (CNN) · **Format:** Python scripts

A CNN-based face recognition system trained to (1) recognize human faces from the **LFW dataset** (13,000+ images) and (2) recognize animal faces from a **Wildlife dataset**, plus verify whether two images are of the same identity.

- **Deliverables:** `train.py`, `predict.py`, `models/` (CNN architecture), `utils/data_loader.py`, `checkpoints/` (saved weights), sample output image showing ~90% target accuracy
- **Key libraries:** torch, torchvision, scikit-learn, Pillow, tqdm
- **Design:** Progressive convolutional layers extract edges → features (eyes, nose, ears) → face structure → identity, culminating in fully-connected classification layers

---

## 🛠️ Tech Stack Across All Projects

| Category | Tools / Libraries |
|---|---|
| Classical ML | scikit-learn, pandas, numpy |
| Deep Learning | TensorFlow / Keras, PyTorch, torchvision |
| Reinforcement Learning | Stable-Baselines3, Gymnasium (Box2D) |
| Generative AI / RAG | LangChain, Google Gemini, ChromaDB |
| Web Frameworks | Flask, Streamlit |
| Deployment | Render, Gunicorn |
| Visualization | Matplotlib, Seaborn |

## ⚙️ General Setup Notes

Each project folder is independent and ships its own `requirements.txt` (where applicable). To run any individual project:

```bash
cd "All Assignments/<project-folder>"
python -m venv venv                 # or: conda create -n <env-name> python=3.10
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A few projects have extra setup steps:
- **Cancer classification** — requires manually downloading the Brain Tumor MRI dataset from Kaggle and placing it in a local `dataset/` folder.
- **RAG-Chatbot-main** — requires a `.env` file with a valid Google Gemini API key.
- **CartPole-agent** — on macOS/Zsh, quote the extras install: `pip install "stable-baselines3[extra]" gymnasium matplotlib`.

Refer to each project's individual `README.md` for full step-by-step instructions, dataset links, and usage examples.

---
