import numpy as np
from sklearn.decomposition import PCA

from ..utils import group_normalize


def run_pca(adata, n_components, convert_to_array=True):
    X = group_normalize(adata.X.toarray() if convert_to_array else adata.X, adata.var.cluster.values, smooth=True)

    intron_clusters = adata.var.cluster.values
    all_intron_clusters = np.unique(intron_clusters)
    first_indices_dict = {}
    for i, c in enumerate(intron_clusters):
        if c not in first_indices_dict:
            first_indices_dict[c] = i
    first_indices = np.array([first_indices_dict[c] for c in all_intron_clusters])
    print("X.shape: ", X.shape)
    X = np.delete(X, first_indices, axis=1)
    print("X.shape: ", X.shape)
    latent = PCA(n_components=n_components).fit_transform(X)
    return latent
