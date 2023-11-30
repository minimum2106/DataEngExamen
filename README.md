Ce repo permet de réaliser une approche séquentielle de réduction de dimension et de clustering (l'un après l'autre) sur des données textuelles.

Les données utilisées proviennent du jeu 20 Newsgroups (disponible dans le package scikit-learn)
L'ensemble de données 20 Newsgroups est une collection d'environ 20 000 documents de groupes de discussion, répartis (presque) uniformément entre 20 groupes de discussion différents.

Les embeddings utilisées sont ceux de sentence-transformers (https://www.sbert.net/)
SentenceTransformers est un framework Python pour l'intégration de phrases, de textes et d'images. Le travail est décrit dans l'article Sentence-BERT : Sentence Embeddings using Siamese BERT-Networks.

Le repo permet de développer un modèle de clustering s'appuyant sur la réduction de la dimensionalité. Les méthodes implémentées sont l'ACP, t-SNE et UMAP. 
Puis, il permet de combiner ce modèle à un algorithme de clustering. La méthode implémentée est k-Means.

Description du contenu:
- le dossier experiments contient les notebooks implémentant les différentes méthodes
- le fichier main.py évalue chacune des approches (ACP+kmeans, AFC+kmeans, UMAP+kmeans) à l'aide des métriques NMI, ARI et Accuracy à partir des classes connues
- le fichier requirements.txt indique les packages requis
