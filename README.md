# Projet MOOC4 : Deep Learning / Reconnaissance caractères

## Prérequis

### Pour Python

Installer les dépendances

```
python3 -m pip install tensorflow keras numpy scipy
```

*Je crois que ça suffit.*

### Pour l'entraînement

Télécharger le dataset

```
git clone https://github.com/kensanata/numbers
```

***Attention** à bien cloner le dépôt dans un dossier `numbers` sinon l'apprentissage ne fonctionnera pas.*

## Fonctionnement

### Pour entraîner le  modèle

```
cd src
python3 train.py
```

*Puis attendre.*


### Pour prédire sur une image personnelle

Placer l'image dans ce dossier (racine) et l'appeler `dessin.png`.

```
cd src
python3 eval.py
# Affiche [0, 0, 0, ..., 0] (liste de 10 éléments avec un 1 en face du nombre, de 0 à 9)
```

Des exemples sont donnés dans `images`. 