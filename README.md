# Faces Dataset Decompositions: A Comparison of Dimensionality Reduction Techniques

This project visually compares several **unsupervised matrix decomposition** (dimensionality reduction) techniques using the classic **Olivetti Faces Dataset** from `scikit-learn`.

It demonstrates how algorithms such as **PCA**, **NMF**, **ICA**, **Sparse PCA**, **Dictionary Learning**, **MiniBatchKMeans**, and **Factor Analysis** extract and represent key features from high-dimensional facial data.

---

## Output
<img width="1920" height="967" alt="LDA vs QDA" src="https://github.com/user-attachments/assets/7dada835-6c20-462f-a481-62bb53395acd" />

---

## Overview

This script provides a **visual and practical comparison** of various decomposition algorithms applied to the same dataset.  
Each technique has its own assumptions and strengths — for example:

- **PCA (Principal Component Analysis)** focuses on maximizing variance.
- **NMF (Non-negative Matrix Factorization)** finds additive, non-negative components.
- **ICA (Independent Component Analysis)** separates statistically independent sources.
- **Sparse PCA** encourages sparsity in the components.
- **Dictionary Learning** builds a sparse, overcomplete basis set.
- **MiniBatchKMeans** treats each cluster center as a “prototype” component.
- **Factor Analysis** models latent factors explaining observed variances.

These methods are compared both visually and quantitatively to help understand how they represent complex face structures.

---

## Implemented Techniques

| Technique | Description | Input Type |
|------------|--------------|-------------|
| **PCA (Randomized SVD)** | Linear projection maximizing variance | Centered data |
| **NMF** | Non-negative factorization of input matrix | Non-negative data |
| **ICA (FastICA)** | Separates independent signals | Centered data |
| **Sparse PCA** | Sparse linear decomposition | Centered data |
| **MiniBatch Dictionary Learning** | Learns sparse basis from data | Centered data |
| **MiniBatch K-Means** | Cluster centers as “representative faces” | Centered data |
| **Factor Analysis (FA)** | Latent variable model explaining observed correlations | Centered data |

Additionally, the script tests **positivity-constrained variants** of Dictionary Learning:
- Positive Dictionary
- Positive Code
- Both Positive Dictionary & Code

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/WallacehCosta/LDA-QDA-Covariance-Ellipsoid-Comparison.git
cd <LDA-QDA-Covariance-Ellipsoid-Comparison>

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install required packages
pip install numpy matplotlib scikit-learn
