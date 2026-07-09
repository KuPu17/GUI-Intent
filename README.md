# 🧠 UI Behavior Engine

An end-to-end multimodal machine learning microservice that translates raw, asynchronous mouse telemetry into psychological friction and workflow intent. 

Instead of relying on traditional analytics like page views or button clicks, this engine calculates the literal kinematics of a user's cursor to detect cognitive load, and uses recurrent neural networks to read spatial-temporal trajectories to forecast their overarching goal.

---

## 🛑 The Problem
Traditional UI analytics are structurally blind to human frustration. They track *where* a user clicked, but not *how* they arrived there. A user who confidently navigates to a checkout page looks identical in traditional analytics to a user who spent 45 seconds wandering the screen in confusion before finally finding the button. 

## ⚡ The Architecture
This project processes raw `(X, Y, Timestamp)` JSON payloads through a dual-branch machine learning architecture to output live behavioral intelligence.

### Branch A: The Kinematic Friction Engine (LightGBM)
Calculates the physical "body language" of the mouse to detect micro-moments of extreme UX friction.
* **Feature Engineering:** Extracts continuous kinematic features including Velocity, Euclidean Distance, Max Hesitation, Tortuosity ($\tau = \frac{L_{path}}{L_{euclidean}}$), and mathematically defines "Rage Clicks."
* **Topological Balancing:** UX friction is rare (20:1 imbalance). Used **SMOTE-Tomek** to mathematically synthesize minority data and delete overlapping boundary points, creating a crisp decision boundary.
* **Leakage Prevention:** Validated using **GroupKFold** cross-validation to mathematically guarantee the model predicts based on universal human kinematics, not memorized user habits.
* **Performance:** Achieved a **0.94 Macro-F1 Score**.

### Branch B: The Multimodal Intent Forecaster (PyTorch GRU)
A recurrent neural network that acts as an NLP engine for spatial UI, predicting one of 6 Macro-Intents (e.g., *E-Commerce Flow*, *Media & Entertainment*).
* **Spatial Tokenizer:** Used a Gaussian Mixture Model (GMM) optimized via Bayesian Information Criterion (BIC) to probabilistically cluster the frontend of the internet into 12 universal semantic zones.
* **Multimodal Fusion:** Fuses discrete topological zones with continuous, log-scaled time-deltas ($\Delta t$) to give the network an understanding of both geometry and rhythm.
* **Stable Focal Loss:** Hand-coded a numerically stable, clamped Focal Loss function to aggressively penalize majority-class overfitting and force the gradient descent to learn rare minority workflows.

---

## 🛠️ Tech Stack & MLOps
* **Core ML:** PyTorch, LightGBM, Scikit-Learn, Imbalanced-Learn
* **Data Processing:** Pandas, NumPy, SciPy
* **Deployment:** FastAPI, Uvicorn
* **Model Hosting (CDN):** Hugging Face Hub (Separation of logic and heavy tensor weights)
---

## 📊 The Dataset (Data Provenance)
The models were trained on the [anaisleila/computer-use-data-psai](https://huggingface.co/datasets/anaisleila/computer-use-data-psai) dataset

---

## 📂 Repository Structure

```text
ui-behavior-engine/
├── src/
│   ├── architectures.py    # Custom PyTorch nn.Modules & Stable Focal Loss
│   └── inference.py        # FastAPI server & HF Model ingestion
├── requirements.txt
└── README.md               # You are here