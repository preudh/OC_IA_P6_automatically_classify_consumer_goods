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
Voici la liste des bibliothèques dans votre environnement Python à installer via pip (anaconda est déconseillé pour gérer
les dépendances) :
```bash
- absl-py 2.1.0
- anyio 4.4.0
- argon2-cffi 23.1.0
- argon2-cffi-bindings 21.2.0
- arrow 1.3.0
- asttokens 2.4.1
- astunparse 1.6.3
- async-lru 2.0.4
- attrs 23.2.0
- Babel 2.14.0
- beautifulsoup4 4.12.3
- bleach 6.1.0
- Brotli 1.1.0
- cached-property 1.5.2
- certifi 2024.8.30
- cffi 1.17.0
- charset-normalizer 3.3.2
- click 8.1.7
- colorama 0.4.6
- comm 0.2.2
- compress_json 1.1.0
- contourpy 1.3.0
- cycler 0.12.1
- debugpy 1.8.5
- decorator 5.1.1
- defusedxml 0.7.1
- entrypoints 0.4
- exceptiongroup 1.2.2
- executing 2.1.0
- fastjsonschema 2.20.0
- filelock 3.16.0
- flatbuffers 24.3.25
- fonttools 4.53.1
- fqdn 1.5.1
- fsspec 2024.9.0
- gast 0.6.0
- gensim 4.3.3
- google-pasta 0.2.0
- grpcio 1.66.1
- h11 0.14.0
- h2 4.1.0
- h5py 3.11.0
- hpack 4.0.0
- httpcore 1.0.4
- httpx 0.27.0
- huggingface-hub 0.24.7
- hyperframe 6.0.1
- idna 3.8
- importlib_metadata 8.4.0
- importlib_resources 6.4.4
- ipykernel 6.29.5
- ipython 8.27.0
- ipywidgets 8.1.5
- isoduration 20.11.0
- jedi 0.19.1
- Jinja2 3.1.4
- joblib 1.4.2
- json5 0.9.25
- jsonpointer 3.0.0
- jsonschema 4.21.1
- jsonschema-specifications 2023.12.1
- jupyter 1.1.1
- jupyter_client 8.6.2
- jupyter-console 6.6.3
- jupyter_core 5.7.2
- jupyter-events 0.10.0
- jupyter-lsp 2.2.4
- jupyter_server 2.13.0
- jupyter_server_terminals 0.5.3
- jupyterlab 4.1.5
- jupyterlab_pygments 0.3.0
- jupyterlab_server 2.25.4
- jupyterlab_widgets 3.0.13
- keras 3.5.0
- kiwisolver 1.4.7
- libclang 18.1.1
- Markdown 3.7
- markdown-it-py 3.0.0
- MarkupSafe 2.1.5
- matplotlib 3.9.2
- matplotlib-inline 0.1.7
- mdurl 0.1.2
- mistune 3.0.2
- ml-dtypes 0.4.0
- namex 0.0.8
- nbclient 0.10.0
- nbconvert 7.16.4
- nbformat 5.10.4
- nest_asyncio 1.6.0
- nltk 3.8.1
- notebook 7.1.2
- notebook_shim 0.2.4
- numpy 1.26.4
- opencv-contrib-python 4.10.0.84
- opencv-python 4.10.0.84
- opt-einsum 3.3.0
- optree 0.12.1
- overrides 7.7.0
- packaging 24.1
- pandas 2.2.2
- pandocfilters 1.5.0
- parso 0.8.4
- pickleshare 0.7.5
- pillow 10.4.0
- pip 24.2
- pkgutil_resolve_name 1.3.10
- platfo

