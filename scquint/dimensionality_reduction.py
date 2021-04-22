from sklearn.decomposition import PCA

from .utils import group_normalize


def run_pca(adata, n_components, convert_to_array=True):
    X = group_normalize(adata.X.toarray() if convert_to_array else adata.X, adata.var.cluster.values, smooth=True)
    latent = PCA(n_components=n_components).fit_transform(X)
    return latent
