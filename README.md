Ce repo permet de réaliser une approche séquentielle de réduction de dimension et de clustering (l'un après l'autre) sur des données textuelles.

Les données utilisées proviennent du jeu 20 Newsgroups (disponible dans le package scikit-learn)
L'ensemble de données 20 Newsgroups est une collection d'environ 20 000 documents de groupes de discussion, répartis (presque) uniformément entre 20 groupes de discussion différents.

Les embeddings utilisées sont ceux de sentence-transformers (https://www.sbert.net/)
SentenceTransformers est un framework Python pour l'intégration de phrases, de textes et d'images. Le travail est décrit dans l'article Sentence-BERT : Sentence Embeddings using Siamese BERT-Networks.

Le repo permet de développer un modèle de clustering s'appuyant sur la réduction de la dimensionalité. Les méthodes implémentées sont l'ACP, t-SNE et UMAP. 
Puis, il permet de combiner ce modèle à un algorithme de clustering. La méthode implémentée est k-Means.

## Description du contenu:
- main.py évalue chacune des approches (ACP+kmeans, t-SNE+kmeans, UMAP+kmeans) à l'aide des métriques NMI, ARI et Accuracy à partir des classes connues
- requirements.txt indique les packages requis
- utils.py comprend des fonctions pour générer des graphiques et comparer les clustesr
- experiment_original_data contient des expérimentations sur les données originales, sans représentation (embeddings) avec t-SNE
- experiments contient des notebooks de travail et d'expérimentation
- labels.pickle contient les étiquettes des données
- embeddings.pickle contient les embeddings de sentence-transformer


## Docker:
- On a généré un image de Docker 

```
docker build -t data_eng .
```

- On a changé le tag 
```
docker tag data_engin yoy888/data_engin
```

- Pour uploader l'image sur DockerHub
```
docker push yoy888/data_engin:latest
```

- Pour monter le volume
```
docker run  -v "/Users/zewei.lin/Downloads/Projet_data_engineering/DataEngExamen":/app --name data_engin data_engin
```