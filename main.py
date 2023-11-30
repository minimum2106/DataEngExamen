from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sentence_transformers import SentenceTransformer
import numpy as np

def dim_red_acp(mat, p):
    pca = PCA(n_components=p)
    
    return pca.fit_transform(mat)  

def dim_red_tsne(mat, p):
    tsne = TSNE(
        n_components = p,  
        learning_rate='auto',      
        init='random'       
        , perplexity=3  
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
        raise Exception("Please select one of the three methods : APC, AFC, UMAP") 


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

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'TSNE', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
