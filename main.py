from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from utils import *
import pickle

# from sentence_transformers import SentenceTransformer
# import numpy as np

def dim_red_acp(mat, p):
    pca = PCA(n_components=p)
    
    return pca.fit_transform(mat)  

def dim_red_tsne(mat, p):
    tsne = TSNE(
        n_components = p,  
        learning_rate='auto',      
        init='random',
        method='exact', 
        perplexity=3  
    )

    return tsne.fit_transform(mat)

def dim_red_umap(mat, p):
    umap = umap.UMAP(random_state=7)
    
    return umap.fit_transform(mat)

def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        return dim_red_acp(mat, p)
        
    elif method=='TSNE':
        return dim_red_tsne(mat, p)
        
    elif method=='UMAP':
        return dim_red_umap(mat, p)
        
    else:
        raise Exception("Please select one of the three methods : APC, t-SNE, UMAP") 


def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(mat)
    
    return kmeans.labels_


if __name__ == "__main__" : 
    # import data
    file = open('embeddings.pickle', 'rb')
    embeddings = pickle.load(file)
    file.close()

    # import labels
    file = open('labels.pickle', 'rb')
    labels = pickle.load(file)
    file.close()


    k = len(set(labels))

    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # embeddings = model.encode(corpus[:2000])


    # Perform dimensionality reduction and clustering for each method
    methods = ['ACP', 'TSNE', 'UMAP']
    for method in methods:
        # Perform dimensionality reduction
        red_emb = dim_red(embeddings, 4, method)

            # Perform clustering
        sum_nmi_score = 0
        sum_ari_score = 0

        # Perform clustering
        for i in range(5):
            pred = clust(red_emb, k)

            # Evaluate clustering results
            sum_nmi_score += normalized_mutual_info_score(pred, labels)
            sum_ari_score += adjusted_rand_score(pred, labels)

        # Print results
        nmi_score = sum_nmi_score / 5
        ari_score = sum_ari_score / 5
        print(f'Method: {method}\nAverage Cross-Validation NMI: {nmi_score:.2f} \nAverage Cross-Validation ARI: {ari_score:.2f}\n')

        # Plot graphs to compare predictions with ground truth 
        print("Visualisation des résultats en 2D")
        clust_viz_2D(embeddings, labels, pred)

        print("Visualisation des résultats en 3D")
        clust_viz_3D(embeddings, labels, pred)
