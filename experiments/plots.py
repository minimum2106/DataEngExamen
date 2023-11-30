import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def clust_viz_2D(embeddings, labels, pred):
  '''
  Fonction permettant de créer des graphiques 2D
  pour comparer les clusters prédits avec les vrais clusters
  '''

  # Dimensions
  dim1 = embeddings[:, 0]
  dim2 = embeddings[:, 1]

  # Création des graphiques des clusters sur 2 dimensions
  plt.figure(figsize = (7, 5))

  # Affichage des clusters prédits
  plt.subplot(1, 2, 1)
  plt.title("Clusters prédits")
  plt.scatter(x=dim1, y=dim2, c=pred, cmap='viridis', alpha=0.2)

  # Affichage des vrais clusters
  plt.subplot(1, 2, 2)
  plt.title("Vrais clusters")
  plt.scatter(x=dim1, y=dim2, c=labels, cmap='viridis', alpha=0.2)
  plt.show()