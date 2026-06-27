# 🔍 Détection d'Anomalies Industrielles - PatchCore sur MVTec AD

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![Docker](https://img.shields.io/badge/Docker-✓-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> 🇬🇧 An English version of this README is available in [`README.md`](README.md)

Détection d'anomalies non supervisée sur des images industrielles avec **PatchCore** (features multiscales ResNet18 + coreset greedy k-center), évalué sur le dataset [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad).

---
## 📋 Table des matières

- [Démonstration](#️-démonstration)
- [Problématique](#-problématique)
- [Méthodes](#-méthodes)
- [Résultats](#-résultats)
- [Structure du projet](#-structure-du-projet)
- [Installation](#️-installation)
- [Utilisation](#-utilisation)
- [Références](#-références)
- [Auteur](#-auteur)

---

## 🖥️ Démonstration

![App screenshot](outputs/figures/app_screenshot.png)

### Comment utiliser l'application

**Étape 1 - Sélectionner une catégorie**
Choisissez le type de pièce industrielle dans le menu déroulant (ex. `bottle`, `capsule`, `screw`).
Le modèle est entraîné sur cette catégorie spécifique - faites toujours correspondre la catégorie à votre image.

**Étape 2 - Uploader une ou plusieurs images**
Vous pouvez uploader **plusieurs images à la fois** (PNG, JPG, JPEG).
Chaque image est analysée indépendamment et les résultats s'affichent l'un après l'autre.

**Étape 3 - Lire les résultats**
Chaque image affiche trois panneaux :
- **Original** - votre image uploadée
- **Carte d'anomalie** - zones rouges = score d'anomalie élevé, bleu = normal
- **Overlay** - carte d'anomalie superposée à l'original pour localiser le défaut

**Étape 4 - Interpréter le score et le verdict**
- **Score d'anomalie** - valeur normalisée dans [0, 1]. Plus élevé = plus probablement défectueux.
- **Verdict** - ✅ Normal ou ❌ Défectueux selon le seuil de détection.
- **Seuil de détection** - ajustable via le slider dans la sidebar. Calibré par catégorie via l'indice de Youden sur le dataset test MVTec. Plus bas = plus sensible (plus de faux positifs). Plus haut = plus conservateur (plus de faux négatifs).

**Étape 5 - Détails techniques**
Dépliez la section *Technical details* pour voir :
- Taille de la memory bank (nombre de patches normaux stockés)
- Méthode de coreset (greedy k-center)
- Normalisation des embeddings (L2)
- Score brut et temps d'inférence

> ⚠️ **Important** : uploadez des images correspondant à la catégorie sélectionnée. Uploader une image de bouteille avec la catégorie capsule sélectionnée produira des résultats non fiables - le modèle n'a pas de référence pour ce type de pièce.

---

## 🎯 Problématique

La détection d'anomalies en inspection industrielle est difficile pour plusieurs raisons :
- **Déséquilibre de classes extrême** - les pièces défectueuses sont rares (~ratio 1:1000)
- **Pas d'exemples anormaux** à l'entraînement (apprentissage one-class)
- **Défauts subtils** - fissures pixel-minces, égratignures, variations de couleur
- **Fonds complexes** - marques d'usinage, textures répétitives

Ce projet compare deux familles d'approches sur MVTec AD :
- **Reconstruction** - Autoencoders convolutifs (V1, V2)
- **Représentation** - PatchCore avec embeddings multiscales ResNet18

---

## 🧠 Méthodes

### Autoencoder (approche par reconstruction)
Entraîner un autoencoder CNN uniquement sur des images normales. Au test, une erreur de reconstruction élevée signale une anomalie. Deux architectures comparées :
- **V1** - encodeur/décodeur simple (3 couches conv)
- **V2** - architecture plus profonde avec décodeur par upsampling

**Limite fondamentale** : un modèle qui reconstruit trop bien va aussi reconstruire les anomalies, ce qui dégrade la détection.

### PatchCore (approche par représentation)
1. Extraire des embeddings de patches multiscales depuis un ResNet18 gelé (layer2 + layer3)
2. Construire une **memory bank** d'embeddings de patches normaux (normalisés L2)
3. Appliquer un **coreset greedy k-center** (10%) pour l'efficacité
4. Au test : calculer la distance au plus proche voisin par patch → carte d'anomalie

---

## 📊 Résultats

### Catégorie Capsule - comparaison des méthodes

| Méthode | AUROC mean | AUROC max | AUROC topk |
|---------|-----------|-----------|------------|
| AE V1 (reconstruction) | 0.398 | 0.627 | 0.556 |
| AE V2 (reconstruction) | 0.563 | 0.613 | 0.615 |
| **PatchCore (représentation)** | **0.876** | **0.941** | **0.938** |

### Benchmark PatchCore - 15 catégories MVTec

![Benchmark AUROC](outputs/figures/benchmark_auroc_15categories.png)

| Catégorie | AUROC (topk) | Seuil optimal |
|-----------|-------------|---------------|
| bottle | 0.999 | 0.351 |
| cable | 0.981 | 0.402 |
| capsule | 0.886 | 0.296 |
| carpet | 0.970 | 0.347 |
| grid | 0.847 | 0.353 |
| hazelnut | 0.999 | 0.419 |
| leather | 0.991 | 0.338 |
| metal_nut | 0.990 | 0.385 |
| pill | 0.870 | 0.368 |
| screw | 0.866 | 0.349 |
| tile | 0.989 | 0.385 |
| toothbrush | 0.989 | 0.367 |
| transistor | 0.945 | 0.397 |
| wood | 0.991 | 0.383 |
| zipper | 0.944 | 0.327 |
| **Moyenne** | **0.957** | - |

Les seuils sont calculés par l'**indice de Youden** (maximise TPR − FPR sur le dataset test).

---

## 📁 Structure du projet

```
industrial-anomaly-detection/
├── api/
│   ├── database.py
│   └── inference.py
    └── main.py
├── app/
│   └── app.py                  # Application Streamlit
├── mvtec_dataset/
│   └── mvtec.py                # Dataset PyTorch MVTec AD
├── models/
│   ├── autoencoder.py          # CNN Autoencoder V1
│   ├── autoencoder_v2.py       # CNN Autoencoder V2
│   ├── patchcore.py            # Pipeline PatchCore
│   └── checkpoints/            # Poids des modèles (.pth)
├── evaluation/
│   └── metrics.py              # AUROC, fonctions de scoring
├── training/
│   └── train_autoencoder.py    # Script d'entraînement (CLI)
├── scripts/
│   └── prepare_memory_banks.py # Précalcul des memory banks coreset
├── utils/
│   └── normalization.py        # Prétraitement des images
├── visualization/
│   └── heatmap.py              # Visualisation des cartes d'anomalie
├── notebooks/
│   ├── 01_autoencoder_experiments.ipynb
│   ├── 02_patchcore_experiments.ipynb
│   ├── 03_embedding_analysis.ipynb
│   ├── 04_multicategory_benchmark.ipynb
│   └── 05_final_comparison.ipynb
├── outputs/
│   ├── figures/                # Graphiques et visualisations
│   └── metrics/                # Résultats JSON
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation

### Prérequis
- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

### Cloner & installer

```bash
git clone https://github.com/Machoudi-1/industrial-anomaly-detection.git
cd industrial-anomaly-detection
poetry install
```

### Dataset

Téléchargez le [dataset MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) et placez-le dans :

```
data/mvtec_ad/
├── bottle/
├── capsule/
├── ...
└── zipper/
```

---

## 🚀 Utilisation

### 1. Entraîner l'autoencoder

```bash
# Entraîner V2 sur capsule (30 epochs par défaut)
poetry run python training/train_autoencoder.py --category capsule --version v2

# Entraîner V1 sur bottle avec paramètres personnalisés
poetry run python training/train_autoencoder.py --category bottle --version v1 --epochs 50 --lr 1e-3
```

### 2. Lancer le benchmark (PatchCore sur 15 catégories)

Ouvrir et exécuter `notebooks/04_multicategory_benchmark.ipynb`.
Mettre `SAVE_MEMORY_BANKS = True` pour sauvegarder les memory banks.

### 3. Préparer les memory banks pour l'app

```bash
# Précalculer les memory banks coreset normalisées L2 (à faire une seule fois)
poetry run python scripts/prepare_memory_banks.py

# Chemins personnalisés (ex: sur Colab)
python scripts/prepare_memory_banks.py \
    --input-dir /chemin/vers/memory_banks \
    --output-dir /chemin/vers/memory_banks/ready
```

### 4. Lancer l'application Streamlit

```bash
poetry run streamlit run app/app.py
```

Ouvrir `http://localhost:8501` dans votre navigateur.

---

### 5. API REST

L'API expose trois endpoints :

- `GET /health` — vérifie que l'API fonctionne
- `POST /predict` — envoie une image et retourne un score d'anomalie et un verdict (Defective ou Normal)
- `GET /history` — récupère les dernières prédictions enregistrées dans la base de données

**Exemple — prédiction depuis un terminal :**

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@data/mvtec_ad/capsule/test/crack/000.png" \
  -F "category=capsule"
```

**Réponse :**

```json
{
  "score": 0.348,
  "verdict": "Defective",
  "threshold": 0.296,
  "inference_time": 0.177
}
```

---

### 6. Docker

**Construire l'image :**

```bash
docker build -t mvtec-anomaly-api .
```

**Lancer le conteneur :**

```bash
docker run -d --name mvtec-api -p 8000:8000 \
  -v $(pwd)/predictions.db:/app/predictions.db \
  mvtec-anomaly-api
```

L'API est ensuite disponible à l'adresse suivante : `http://localhost:8000`.

**Arrêter le conteneur :**

```bash
docker stop mvtec-api
```
## 📚 Références

- **PatchCore** : Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022. [arxiv](https://arxiv.org/abs/2106.08265)
- **MVTec AD** : Bergmann et al., *MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- **Focal Loss** : Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- **BAGAN** : Mariani et al., *BAGAN: Data Augmentation with Balancing GAN*, 2018.

---
## 👤 Auteur

**Machoudi ADEGOUNTE** — Mathématiques Appliquées & Machine Learning