#  Anomaly Detection with Autoencoder (MVTec AD)

##  Project Overview

This project focuses on **unsupervised anomaly detection** using an **autoencoder** trained on the MVTec AD dataset.

The objective is to detect defects in industrial images by learning to reconstruct normal samples and identifying anomalies through reconstruction errors.

---

## 📂 Dataset

* Dataset: **MVTec Anomaly Detection**
* Category used: `capsule`
* Training: only **normal images**
* Testing: **normal + anomalous images**

---

##  Methodology

### 1. Model

A convolutional **Autoencoder** is trained to reconstruct input images.

* Input: RGB images (resized)
* Loss: Mean Squared Error (MSE)
* Training: 30 epochs

---

### 2. Anomaly Scoring

We evaluate different strategies to convert reconstruction errors into anomaly scores:

* **Mean error** (global average)
* **Max error** (maximum pixel error)
* **Top-k mean error** (average of highest k% errors)

---

### 3. Evaluation

* Score distributions (train / test normal / anomalies)
* ROC Curve
* AUROC metric

---

##  Results

| Method     | AUROC |
| ---------- | ----- |
| Mean       | 0.499 |
| Max        | 0.530 |
| Top-k Mean | 0.535 |

---

##  Analysis

* The autoencoder successfully reconstructs the **global structure** of images.
* However, it also reconstructs **anomalies**, reducing discrimination power.
* Global averaging (mean) dilutes local defects.
* Top-k scoring improves results slightly but remains insufficient.

 **Key insight**:
The limitation comes from the **model capacity**, not only from the scoring method.

---

##  Limitations

* Simple autoencoder generalizes too well
* Poor sensitivity to small/local anomalies
* Reconstruction-based methods struggle with subtle defects

---

##  Future Improvements

* Improve architecture (e.g., UNet-like autoencoder)
* Use perceptual losses (SSIM)
* Try state-of-the-art methods:

  * PatchCore
  * PaDiM
  * FastFlow

---

##  Tech Stack

* Python
* PyTorch
* NumPy / Matplotlib
* Scikit-learn (ROC, AUROC)

---

##  Key Takeaways

* Reconstruction quality ≠ anomaly detection performance
* Scoring strategy matters, but model design is critical
* Local anomaly detection requires specialized approaches

---

## Author

Machoudi ADEGOUNTE
Master’s student in Applied Mathematics & AI
