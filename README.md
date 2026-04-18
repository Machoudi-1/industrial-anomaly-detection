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

## Methods

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

## Feature Space Analysis (PCA & t-SNE)

To better understand how each method represents the data, we analyzed the learned feature spaces using **PCA (global structure)** and **t-SNE (local structure)**.

### Objective

The goal of this analysis is **not to directly separate anomalies**, but to:

- Evaluate the **structure of the feature space**
- Assess **generalization (train vs test)**
- Understand whether anomalies create **local deviations**

---

## PatchCore Feature Space

### Observations

#### PCA (Global structure)

- The feature space is **well-structured**
- Training and test samples follow a **similar global distribution**
- Multiple **sub-clusters** appear, corresponding to different visual patterns
- Anomalies are **not globally separated**, but remain embedded in the same structure

#### t-SNE (Local structure)

- Clear **local clusters** are visible
- Each cluster corresponds to **specific image patterns or regions**
- Anomalies are **mixed within clusters**, but tend to appear as **local deviations**

#### Score Distribution

- Strong separation between:
  - **Normal samples → low scores**
  - **Anomalies → higher scores**
- Minimal overlap between the two distributions
- Confirms high anomaly detection performance

---

### Interpretation

- The feature extractor learns a **rich and structured representation**
- The model generalizes well to unseen data
- **Anomaly detection is not based on global separation**, but on:

> **local distance to normal patterns (nearest neighbors)**

---

### Hypothesis

> A well-structured feature space enables effective anomaly detection, even without explicit separation between normal and anomalous samples.

---

## Autoencoder Feature Space

### Observations

#### PCA

- The latent space is **diffuse and poorly structured**
- No clear clustering or organization
- Test samples do not align clearly with training distribution

#### t-SNE

- No meaningful clusters emerge
- Data points are scattered randomly
- No clear pattern differentiation

#### Score Distribution

- Strong overlap between normal and anomalous samples
- Poor separability
- Low AUROC

---

### Interpretation

- The autoencoder fails to learn a **discriminative representation**
- It focuses on **global reconstruction**, not fine details
- Anomalies are often **reconstructed**, reducing detection ability

---

### Hypothesis

> Reconstruction-based methods struggle to detect subtle anomalies because they tend to generalize and reconstruct both normal and abnormal patterns.

---

## Comparative Insight

| Property                | Autoencoder | PatchCore |
|------------------------|------------|----------|
| Feature structure      | Diffuse     | Structured |
| Local clusters         | No          | Yes |
| Generalization         | Weak        | Strong |
| Anomaly separation     | Poor        | Strong (via scores) |
| Detection mechanism    | Reconstruction | Nearest neighbor |

---

## Key Takeaway

> **A structured feature space is more important than reconstruction quality for anomaly detection.**

PatchCore succeeds because it leverages **local feature consistency**, while autoencoders fail due to **over-generalization**.

---

## Future Work

- Evaluate all MVTec categories
- Optimize PatchCore (better coreset, normalization, FAISS)
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