# SY32_TD2_Groupe_D
Projet de vision par ordinateur

## Installation de l'environnement

Pour installer l'environnement conda nécessaire pour exécuter les codes de ce projet, veuillez suivre les étapes suivantes :

1. Téléchargez et installez [Anaconda](https://www.anaconda.com/products/individual) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) si ce n'est pas déjà fait.
2. Créez un nouvel environnement conda en utilisant le fichier `environment.yml` fourni dans ce dépôt :
```bash
conda env create -f environment.yml
```
1. Activez l'environnement nouvellement créé :
```bash
conda activate utc-sy32
```
1. (Facultatif) Si vous souhaitez utiliser un noyau Jupyter spécifique à cet environnement, installez ipykernel :
```bash
conda install -c anaconda ipykernel
```
Puis enregistrez le noyau :
```bash
python -m ipykernel install --user --name=utc-sy32
```

Vous êtes maintenant prêt à exécuter les codes de ce projet dans l'environnement `utc-sy32`.


## Machine Learning

Pour pouvoir prédire sur de nouvelles données et que vous n'avez pas encore le modèle enregistrer veuillez suivre ces étapes :
1. Lancer le script `transform_dataset.py`présent dans le dossier utils afin d'obtenir le dossier `full_datset` contenant toutes les classes de panneaux
2. Executer l'entiereté des fichiers .ipynb du dossier `machine_learning`
3. Eventuellement changer le nom et le chemin du dossier à prédire

Sinon vous pouvez n'executer que le début permettant de récupérer les variables et les fonctions utiles puis lancer la prédiction


## Deep Learning

### Final implementation

Contient les codes de l'implémentation finale de notre modèle de détection en deep learning.

Pour lancer la détection sur les images de test. Faire cette commande depuis le repertoire racine du projet : 
```
python ./deep_learning/final_implemenation/run_on_test.py
```

  
**Réentrainement :**   
Pour lancer l'entrainement de U-net
```
python ./deep_learning/final_implemenation/train_segmentation_model.py
```
Pour lancer l'entrainement du classifieur
```
python ./deep_learning/final_implemenation/train_classification_model.py
```
  
Les poids de ces réseaux vont s'enregistrer dans le repertoire checkpoints  
La définition des modèles est dans le repertoire models

### YOLO implementation
Code de la tentative d'implémentation de YOLO (demande puissance de calcul trop importante -> Kaggle)

### YOLO fine tuning
Code du fine tuning

Dans le fichier run.ipynb, saisir le chemin d'une image et visualiser la détection réalisé de YOLO sur celle ci.

**Réentrainement :**  
Transformer le dataset au format accepté par YOLO via le notebook creation_dataset.ipynb  
Réentrainer le modele sur ce dataset avec train_yolo.ipynb  



## Auteurs
- [Martin C.](github.com/martincrz)
- [Sacha S.](github.com/sacha-sz)