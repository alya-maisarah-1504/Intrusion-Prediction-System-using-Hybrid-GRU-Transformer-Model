# 🛡️ DoH Intrusion Prediction System

Hybrid GRU-Transformer deep learning model for classifying encrypted **DNS-over-HTTPS (DoH)** network traffic, with a live Streamlit dashboard for real-time prediction and packet capture.

Final Year Project — Bachelor of Computer Science (Hons.), Universiti Teknologi MARA (UiTM)

---

## 📌 Overview

DNS-over-HTTPS encrypts DNS queries inside regular HTTPS traffic, which protects user privacy but also lets attackers hide malicious DNS tunneling activity from traditional network monitoring. It classifies network flow sequences into four categories — **DoH**, **NonDoH**, **Benign**, and **Malicious** — using a custom Hybrid GRU-Transformer architecture trained on real captured traffic.

## 🧠 Model Architecture

A dual-branch deep learning model that fuses recurrent and attention-based feature extraction:

- **GRU branch:** 3× stacked Bidirectional GRU (256→128→64 units) + BatchNorm + Dropout + multi-head self-attention
- **Transformer branch:** 4× Transformer encoder blocks (embed_dim=128, 8 heads, feed-forward=512)
- **Cross-attention fusion:** GRU and Transformer pooled representations attend to each other bidirectionally
- **Classification head:** Dense 512→256→128→64→4 (softmax)
- **Total parameters:** 2,540,356

## 📊 Results (locked, verified)

| Model | Accuracy | FAR | Macro AUC | Mean AP |
|---|---|---|---|---|
| GRU | 78.64% | 11.49% | 0.9366 | 0.8186 |
| Transformer | 78.00% | 14.53% | 0.9323 | 0.8012 |
| **Hybrid GRU-Transformer** | **79.14%** | **11.87%** | **0.9382** | **0.8229** |
| Ensemble (GRU+Hybrid) | 79.09% | 11.57% | 0.9380 | 0.8222 |

Per-class recall (Heavy Hybrid): DoH 64.38% · NonDoH 100.00% · Benign 100.00% · Malicious 52.20%

**Dataset:** Combined Hokkaido CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD + BCCC-CIRA-CIC-DoHBrw-2020 — 200,000 sequences (50k/class), 36 engineered flow features, sequence length 10, 60/20/20 train/val/test split.

## 🖥️ Dashboard

A Streamlit dashboard with three prediction modes:

1. **Dataset Simulation** — streams real labeled test samples through the model one by one, with live accuracy tracking
2. **Live Network Capture** — real-time packet sniffing (via Scapy/Npcap) with live inference on captured flows

Plus Model Comparison, Dataset Info, and About pages.

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `Streamlit` · `Scapy` · `Plotly` · `scikit-learn` · `Google Colab` (training)

## 🚀 Running Locally

```bash
git clone https://github.com/alya-maisarah-1504/Intrusion-Prediction-System-using-Hybrid-GRU-Transformer-Model.git
cd Intrusion-Prediction-System-using-Hybrid-GRU-Transformer-Model
pip install -r requirements.txt
streamlit run app.py
```

**Requirements:** trained model weights (`models/hybrid_weights_cpu.npz`) and scaler (`models/scaler_FINAL.pkl`) must be present — see `models/` folder. For live capture mode (Mode 3) on Windows, [Npcap](https://npcap.com/) must be installed and the app run with administrator privileges.

## 📁 Repository Structure

```
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── models/
│   ├── hybrid_weights_cpu.npz   # trained model weights
│   └── scaler_FINAL.pkl         # MinMaxScaler (36 features)
└── data/
    ├── X_te.npy             # test sequences
    └── y_te.npy             # test labels
```

## 📄 License

Academic project — Universiti Teknologi MARA, 2026.
