# Classification de commentaires YouTube

## Présentation

Ce projet porte sur la **classification binaire de commentaires YouTube (spam/pas spam)**.

L'objectif est de prédire si la variable `CLASS` à partir du contenu du commentaire et de quelques informations complémentaires comme l'auteur, la vidéo et la date de publication est un spam ou pas. 

Le projet a été réalisé en Python avec principalement **pandas, scikit-learn, NumPy, SciPy et Matplotlib**.

L'approche retenue dans ce notebook est volontairement progressive : commencer par une méthode classique et facilement interprétable avant d'envisager des modèles plus complexes.

## Problématique

La variable cible `CLASS` est binaire. Le problème est donc traité comme un problème de **classification supervisée**.

La démarche est la suivante :

1. charger et vérifier les données ;
2. nettoyer les commentaires ;
3. créer quelques variables simples ;
4. transformer le texte avec **TF-IDF** ;
5. combiner les informations textuelles et numériques ;
6. entraîner une **régression logistique** ;
7. comparer plusieurs valeurs du paramètre `C` avec une validation croisée ;
8. évaluer le meilleur modèle sur un jeu de validation ;
9. réentraîner le modèle sur l'ensemble des données d'entraînement ;
10. produire les prédictions du fichier de test.

## Données
Le projet a été organisé dans le cadre d'une compétition privée sur Kaggle. Donc les fichiers csv ne peuvent être fournis.

Le notebook attend deux fichiers CSV placés dans le même dossier :

```text
train_yt.csv
test_yt.csv
```

Le jeu d'entraînement contient notamment la variable cible `CLASS` ainsi que les informations utilisées pour la classification. Le jeu de test est utilisé uniquement pour générer les prédictions finales.

Les commentaires sont principalement en anglais, ce qui explique l'utilisation de la liste de mots vides anglaise ainsi que `stop_words="english"` dans le vectoriseur TF-IDF.

## Prétraitement du texte

Une fonction de nettoyage est utilisée pour :

- passer le texte en minuscules ;
- retirer les liens ;
- retirer les mentions ;
- normaliser les espaces ;
- supprimer quelques mots très fréquents ;
- conserver une représentation simple du commentaire.

Le but est de nettoyer le texte sans appliquer de traitement excessif qui pourrait supprimer une information utile.

## Variables utilisées

En plus du texte, plusieurs caractéristiques simples sont construites :

- longueur du commentaire ;
- proportion de lettres majuscules ;
- nombre de signes de ponctuation ;
- heure de publication ;
- jour de la semaine ;
- mois de publication ;
- fréquence de l'auteur ;
- fréquence de la vidéo.

Les fréquences de l'auteur et de la vidéo sont calculées uniquement sur les données d'entraînement lors de la phase de validation afin d'éviter d'utiliser indirectement des informations du jeu de validation.

## Représentation TF-IDF

Les commentaires sont transformés avec `TfidfVectorizer`.

Le modèle utilise :

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    stop_words="english"
)
```

Les unigrammes et bigrammes sont utilisés afin de conserver à la fois les mots seuls et certaines petites expressions.

## Modèle

Le modèle retenu est une **régression logistique** avec :

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

La régression logistique est adaptée à une classification binaire et constitue une première approche simple et rapide pour des données représentées avec TF-IDF.

## Validation et choix du paramètre `C`

Plusieurs valeurs de `C` sont testées avec `GridSearchCV` :

```text
0.1, 0.5, 1, 2, 5
```

Une validation croisée en 3 parties est utilisée sur le jeu d'entraînement interne.

Le jeu de validation final reste séparé pendant cette recherche afin d'avoir une évaluation indépendante après le choix du paramètre.

## Évaluation

Le notebook utilise notamment :

- **Accuracy** ;
- **ROC-AUC** ;
- **rapport de classification** ;
- **matrice de confusion**.

L'accuracy donne une vision globale des prédictions correctes, tandis que la matrice de confusion permet de distinguer les différents types d'erreurs.

## Prédictions finales

Après sélection du meilleur paramètre `C`, le modèle est réentraîné sur l'ensemble de `train_yt.csv`.

Les prédictions sur `test_yt.csv` sont enregistrées dans :

```text
predictions.csv
```

avec les colonnes :

```text
ID
y_pred
```

## Installation

Installer les bibliothèques nécessaires :

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter
```

Puis lancer Jupyter :

```bash
jupyter notebook
```

Ouvrir ensuite le notebook et exécuter les cellules dans l'ordre.



## Ce que j'ai appris

Ce projet m'a surtout permis de travailler sur :

- le nettoyage de données textuelles ;
- la représentation d'un texte avec TF-IDF ;
- la construction de variables simples à partir de données textuelles et temporelles ;
- la classification binaire avec scikit-learn ;
- la validation croisée ;
- l'évaluation avec plusieurs métriques ;
- la construction d'une pipeline de prédiction reproductible.
