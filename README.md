# Industrial Anomaly Detection (MVTec AD)

## Project Overview

This project explores **unsupervised anomaly detection** methods for industrial inspection using the **MVTec AD dataset**.

Two main approaches are investigated:

- **Reconstruction-based methods** (Autoencoders)
- **Feature-based methods** (PatchCore-inspired pipeline)

The goal is to understand their behavior, compare their performance, and build a robust anomaly detection system.

---

## Dataset

- Dataset: **MVTec Anomaly Detection**
- Categories evaluated:
  - `capsule`
  - `bottle`
  - `screw`

Training is performed using **only normal samples**.  
Testing includes **both normal and anomalous images**.

---

##  Methods

### 1. Autoencoder

Two architectures were implemented:

- **Autoencoder V1**: simple convolutional model
- **Autoencoder V2**: deeper encoder + upsampling decoder

#### Anomaly scoring:
- Mean error
- Max error
- Top-k mean error

---

### 2. PatchCore (Feature-based)

A simplified **PatchCore pipeline** was implemented using a pretrained **ResNet18**.

#### Pipeline:

1. Feature extraction (CNN backbone)
2. Patch embedding extraction
3. Memory bank (normal features)
4. Nearest neighbor search
5. Anomaly maps + image-level scores

#### Improvements:
- Multi-scale features (**layer2 + layer3**)
- Coreset sampling (random / greedy)
- Visualization (heatmaps + overlays)

---

## Results

### Autoencoder (Capsule category)

| Method     | AUROC |
| ---------- | ----- |
| Mean       | 0.499 |
| Max        | 0.530 |
| Top-k Mean | 0.535 |

---

### PatchCore

#### Capsule
- Significant improvement over autoencoder

#### Bottle
- Mean: **0.9857**
- Max: **0.9992**
- Top-k: **1.0000**

#### Screw
- Mean: **0.8225**
- Max: **0.9746**
- Top-k: **0.9813**

---

## Key Insights

### 1. Autoencoder limitations
- Good global reconstruction
- Poor local anomaly detection
- Reconstruction quality ≠ detection performance

---

### 2. PatchCore strengths
- Strong performance on all tested categories
- Excellent detection of localized anomalies
- Robust to complex textures

---

### 3. Importance of scoring strategy
- Mean → weak (dilutes anomalies)
- Max → strong
- Top-k → best trade-off

---

### 4. Category-dependent difficulty
- `bottle` → easy (simple structure)
- `screw` → harder (fine details, textures)

---

## Future Work

- Evaluate all MVTec categories
- PCA / t-SNE analysis of embeddings
- Improve PatchCore (better coreset, normalization)
- Implement other methods:
  - PaDiM
  - FastFlow
- Build a **Streamlit demo** for interactive anomaly detection

---

## Tech Stack

- Python
- PyTorch
- torchvision
- NumPy / Matplotlib
- Scikit-learn

---

## Author

Machoudi ADEGOUNTE  
Master’s student in Applied Mathematics, Data Science & AI