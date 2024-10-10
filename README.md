### README: Classification Automatique des Biens de Consommation

#### Aperçu du Projet :
Ce projet a pour objectif de développer un système de classification automatique des biens de consommation en utilisant
à la fois les descriptions textuelles et les images des produits. L'objectif principal est d'améliorer l'expérience 
utilisateur en automatisant l'attribution des catégories de produits sur une marketplace. Cela est réalisé à l'aide de
l'apprentissage supervisé pour la classification des images et de techniques non supervisées pour l'extraction de
caractéristiques et le clustering.

#### Étapes Principales :

1. **Prétraitement des Données** :
   - **Données Textuelles** : Nettoyage et prétraitement des descriptions des produits (Tokenisation, TF-IDF,
   - Embeddings Word2Vec, BERT).
   - **Données Images** : Redimensionnement, normalisation, et prétraitement des images en utilisant des techniques
   - standard (ex. modèle pré-entraîné VGG-16).

2. **Extraction de Caractéristiques** :
   - **Caractéristiques Textuelles** : Extraction de caractéristiques à partir des descriptions textuelles en utilisant
   - TF-IDF et les embeddings (Word2Vec, BERT).
   - **Caractéristiques Visuelles** : Extraction des caractéristiques profondes des images avec VGG-16 (CNN).

3. **Apprentissage Non Supervisé** :
   - Application du clustering (K-Means) sur les caractéristiques extraites afin d'évaluer le regroupement naturel
   - des produits sans utiliser les labels. 

4. **Apprentissage Supervisé** :
   - Classification des images dans des catégories prédéfinies en utilisant un modèle CNN basé sur VGG-16, adapté à nos
   - catégories grâce à des couches supplémentaires.

#### Livrables :
- Notebooks pour le prétraitement des données et l'extraction des caractéristiques.
- Notebook pour la classification supervisée des images.
- Script Python pour tester une API et extraire les produits dans un fichier CSV.
- Présentation détaillant l’étude de faisabilité et les résultats de la classification supervisée.

#### Technologies Utilisées :
- **Langages** : Python
- **Librairies** : TensorFlow, Keras, Scikit-Learn, NLTK, Gensim, OpenCV.
- **Outils** : Jupyter Notebook.
- **Modèles** : Word2Vec, BERT, VGG-16, K-Means, SIFT, TF-IDF.
 
#### Bibliothèques :
Voici la liste des principales bibliothèques dans votre environnement Python à installer via pip (anaconda est déconseillé pour gérer
les dépendances) :

### Bibliothèques essentielles :
- **numpy** : Bibliothèque fondamentale pour le calcul numérique avec Python.
- **pandas** : Outil clé pour la manipulation et l'analyse des données, notamment pour les DataFrames.
- **scikit-learn** : Bibliothèque essentielle pour les algorithmes de machine learning.
- **tensorflow** et **keras** : Utilisés pour créer et entraîner des modèles de deep learning.
- **matplotlib** et **seaborn** : Bibliothèques pour la visualisation de données et de graphiques.
- **opencv-python** et **opencv-contrib-python** : Pour la manipulation et le traitement d'images.
- **nltk** : Bibliothèque pour le traitement du langage naturel.
- **huggingface-hub** : Pour utiliser des modèles pré-entraînés dans le traitement du langage naturel.
  
### Outils pour notebooks :
- **jupyter** : Environnement de travail interactif pour les notebooks.
- **ipython** : Shell interactif amélioré qui supporte Jupyter.
- **ipywidgets** : Ajout de widgets interactifs dans les notebooks Jupyter.
  
### Autres utilitaires :
- **joblib** : Utilitaire pour la parallélisation et la persistance des modèles.
- **h5py** : Utilisé pour manipuler les fichiers HDF5, souvent en conjonction avec TensorFlow/Keras.
- **pillow** : Manipulation et traitement d'images.
- **requests** et **httpx** : Bibliothèques pour les requêtes HTTP et l'interaction avec des API.
- **beautifulsoup4** : Pour le scraping web et l'analyse des documents HTML.
- **sqlalchemy** : Gestion des bases de données relationnelles.

Ces bibliothèques sont essentielles pour les tâches courantes de **machine learning**, de **traitement d'images** et
de **data science**.
