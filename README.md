# SY32 - Détection de Panneaux de Signalisation

Ce dépôt regroupe l'ensemble des travaux réalisés dans le cadre de l'UV **SY32** à l'UTC, dédiée à la **vision par ordinateur**.

Le projet porte sur la **détection et la classification de panneaux de signalisation routière** à partir d'un dataset annoté d'images. Deux grandes approches ont été explorées : le machine learning classique et le deep learning.

<br/>

## Vue d'ensemble

| Approche | Méthode | Dossier |
|----------|---------|---------|
| Machine Learning | Descripteurs HOG + couleur, fenêtre glissante | [`machine_learning/`](machine_learning/) |
| Deep Learning | U-Net (segmentation) + classifieur CNN | [`deep_learning/final_implementation/`](deep_learning/final_implementation/) |
| Deep Learning | Fine-tuning YOLOv8 | [`deep_learning/yolo_fine_tuning/`](deep_learning/yolo_fine_tuning/) |

<br/>

## Dataset

Le dataset est organisé en trois partitions : entraînement, validation et test. Les annotations sont fournies au format CSV (coordonnées des boîtes englobantes).

| Partition | Dossier |
|-----------|---------|
| Entraînement | [`dataset/train/`](dataset/train/) |
| Validation | [`dataset/val/`](dataset/val/) |
| Test | [`dataset/test/`](dataset/test/) |

**Classes détectées :**

| Classe | Description |
|--------|-------------|
| `frouge` | Feu rouge |
| `forange` | Feu orange |
| `fvert` | Feu vert |
| `stop` | Panneau stop |
| `ceder` | Cédez le passage |
| `interdiction` | Panneau d'interdiction |
| `danger` | Panneau de danger |
| `obligation` | Panneau d'obligation |

<br/>

## Machine Learning

Deux notebooks explorent des approches classiques de détection :

| Notebook | Méthode |
|----------|---------|
| [`ml_color_hog.ipynb`](machine_learning/ml_color_hog.ipynb) | Descripteurs couleur et HOG pour la classification |
| [`ml_sliding_window.ipynb`](machine_learning/ml_sliding_window.ipynb) | Détection par fenêtre glissante |

**Pour lancer la prédiction sur de nouvelles données :**

1. Exécuter [`utils/transform_dataset.py`](utils/transform_dataset.py) pour générer le dossier `full_dataset` contenant toutes les classes.
2. Exécuter les notebooks du dossier `machine_learning/` dans l'ordre.
3. Adapter si besoin le chemin du dossier à prédire en fin de notebook.

<br/>

## Deep Learning

### Implémentation finale

Contient les codes de l'implémentation finale du pipeline de détection en deep learning : segmentation par U-Net suivie d'une classification CNN.

**Détection sur les images de test** (depuis la racine du projet) :

```bash
python ./deep_learning/final_implementation/run_on_test.py
```

**Réentrainement :**

```bash
# Entrainement du modèle de segmentation U-Net
python ./deep_learning/final_implementation/train_segmentation_model.py

# Entrainement du classifieur
python ./deep_learning/final_implementation/train_classification_model.py
```

Les poids des réseaux sont sauvegardés dans [`checkpoints/`](deep_learning/final_implementation/checkpoints/).
La définition des architectures est dans [`models/`](deep_learning/final_implementation/models/).

| Fichier | Rôle |
|---------|------|
| [`config.py`](deep_learning/final_implementation/config.py) | Paramètres globaux (tailles, classes, batch size) |
| [`load_data.py`](deep_learning/final_implementation/load_data.py) | Chargement et préparation des données |
| [`detection_tools.py`](deep_learning/final_implementation/detection_tools.py) | Outils de post-traitement et d'affichage |
| [`train_segmentation_model.py`](deep_learning/final_implementation/train_segmentation_model.py) | Entrainement U-Net |
| [`train_classification_model.py`](deep_learning/final_implementation/train_classification_model.py) | Entrainement du classifieur |
| [`run_on_test.py`](deep_learning/final_implementation/run_on_test.py) | Inférence sur les images de test |

---

### Fine-tuning YOLOv8

Tentative d'adaptation de YOLOv8 au dataset du projet.

**Utilisation :**

1. Convertir le dataset au format YOLO via [`creation_dataset.ipynb`](deep_learning/yolo_fine_tuning/creation_dataset.ipynb).
2. Lancer l'entrainement avec [`train_yolo.py`](deep_learning/yolo_fine_tuning/train_yolo.py).
3. Visualiser une détection sur une image dans [`run.ipynb`](deep_learning/yolo_fine_tuning/run.ipynb).

> Note : l'entrainement complet de YOLO nécessite une puissance de calcul importante (Kaggle recommandé).

<br/>

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/sacha-sz/UTC-SY32.git
cd UTC-SY32

# Créer et activer l'environnement conda
conda env create -f environment.yml
conda activate utc-sy32

# (Facultatif) Enregistrer un noyau Jupyter dédié
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=utc-sy32
```

<br/>

## Technologies utilisées

- **Python 3** - langage principal
- **PyTorch** - entrainement des modèles deep learning (U-Net, classifieur)
- **Ultralytics YOLOv8** - détection par fine-tuning
- **Scikit-learn** - pipelines machine learning
- **OpenCV** - traitement d'images, fenêtre glissante, HOG
- **Pandas / NumPy** - manipulation des données et annotations
- **Matplotlib** - visualisations
- **Jupyter Notebook** - environnement d'analyse et d'expérimentation

<br/>

## Licence

Ce projet est distribué sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus d'informations.

<br/>

## Auteurs

- **[@martincrz](https://github.com/martincrz)**
- **[@sacha-sz](https://github.com/sacha-sz)**