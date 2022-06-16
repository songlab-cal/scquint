import numpy as np
from sklearn.decomposition import PCA

from ..data import group_normalize


def run_pca(adata, n_components, convert_to_array=True):
    X = group_normalize(adata.X.toarray() if convert_to_array else adata.X, adata.var.intron_group.values, smooth=True)

    intron_groups = adata.var.intron_group.values
    all_intron_groups = np.unique(intron_groups)
    first_indices_dict = {}
    for i, c in enumerate(intron_groups):
        if c not in first_indices_dict:
            first_indices_dict[c] = i
    first_indices = np.array([first_indices_dict[c] for c in all_intron_groups])
    X = np.delete(X, first_indices, axis=1)
    latent = PCA(n_components=n_components).fit_transform(X)
    return latent
