# 🐱 Image Compression with Tucker and FFT

## 📌 Project Overview
This project explores **image compression and reconstruction** using tensor decomposition.  
Two approaches are compared:  

1. **Spatial Tucker Decomposition** – direct tensor compression.  
2. **FFT-Tucker Decomposition** – applying Fast Fourier Transform (RFFT) before Tucker.  

👉 The main research question: *Does FFT-Tucker provide better reconstructions than standard Tucker?*

---

## ⚡ Key Features
- 📉 **100x compression** (100 MB → ~1 MB).  
- 🎯 **Over 99% reconstruction accuracy**.  
- ⚙️ **GPU-accelerated** computations with PyTorch.  
- 🧩 Attempted **UMAP & SOM clustering** for dataset cleaning.  
- ☁️ Built a **JupyterHub server** with GPU passthrough.  

---

## 📊 Results
- **Spatial Tucker** → simpler and faster, but loses fine details.  
- **FFT-Tucker** → more accurate, especially in high-frequency details.  

---

## 🧮 Mathematical Background
- **Tucker Decomposition**  
  Compresses a high-dimensional tensor into a smaller **core tensor** and **factor matrices**.  

- **Mode-n Unfolding**  
  Reshapes a tensor along one dimension, enabling matrix operations.  

- **FFT (Fast Fourier Transform)**  
  Transforms data into the frequency domain.  
  With **RFFT**, only half the spectrum is stored thanks to symmetry → memory savings.  

---

## 📂 Dataset
- ~**900 cat images** 🐱.  
- Highly contextual and visually similar, which made clustering very challenging.  

---

## 🧪 Clustering Attempts
- **UMAP** – nonlinear dimensionality reduction.  
- **SOM (Self-Organizing Maps)** – neural-network-based clustering.  
- ❌ Both methods failed to separate images clearly due to similarity.  
- ✅ Images were grouped manually in the end.  

---
