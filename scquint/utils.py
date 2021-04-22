from collections import Counter

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def make_cluster_summation_cpu(intron_clusters):
    n_introns = len(intron_clusters)
    n_clusters = len(np.unique(intron_clusters))
    rows, cols = zip(*list(enumerate(intron_clusters)))
    vals = np.ones(n_introns, dtype=int)
    cluster_summation = sp_sparse.coo_matrix(
        (vals, (rows, cols)), (n_introns, n_clusters)
    ).tocsr()
    return cluster_summation


def relabel(labels):
    all_old_labels = pd.unique(labels).tolist()
    mapping = {c: i for i, c in enumerate(all_old_labels)}
    new_labels = np.array([mapping[l] for l in labels])
    return new_labels


def group_normalize(X, groups, smooth=False):
    print("assuming np.array for the moment")
    cluster_summation = make_cluster_summation_cpu(groups)
    if smooth:
        intron_sum = X.sum(axis=0)
        cluster_sum = intron_sum @ cluster_summation
        X = X + intron_sum / cluster_sum[groups]
    cluster_sums = X @ cluster_summation
    return X / cluster_sums[:,groups]


def filter_min_cells_per_feature(adata, min_cells_per_feature, idx_cells_to_count=slice(None)):
    print("filter_min_cells_per_feature")
    idx_features = np.where((adata.X[idx_cells_to_count] > 0).sum(axis=0).A1 >= min_cells_per_feature)[0]
    adata = adata[:, idx_features]
    # adata.var = adata.var.reset_index(drop=True)
    adata.var.cluster = relabel(adata.var.cluster.values)
    adata = filter_singletons(adata)
    return adata


def filter_min_cells_per_cluster(adata, min_cells_per_cluster, idx_cells_to_count=slice(None)):
    print("filter_min_cells_per_cluster")
    clusters = adata.var.cluster.values
    cluster_summation = make_cluster_summation_cpu(clusters)
    n_cells_per_cluster = ((((adata.X[idx_cells_to_count]) @ cluster_summation) > 0).sum(axis=0)).A1
    idx_clusters = np.where(n_cells_per_cluster >= min_cells_per_cluster)[0]
    idx_features = np.where(np.isin(clusters, idx_clusters))[0]
    adata = adata[:, idx_features]
    # adata.var = adata.var.reset_index(drop=True)
    adata.var.cluster = relabel(adata.var.cluster.values)
    adata = filter_singletons(adata)
    return adata


def filter_singletons(adata):
    print("filter_singletons")
    cluster_counter = Counter(adata.var.cluster.values)
    cluster_counts = np.array([cluster_counter[c] for c in adata.var.cluster.values])
    idx_features = np.where(cluster_counts > 1)[0]
    adata = adata[:, idx_features]
    # adata.var = adata.var.reset_index(drop=True)
    adata.var.cluster = relabel(adata.var.cluster.values)
    return adata


def _run_differential_expression(adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene):
    import scanpy as sc
    cell_idx_all = np.concatenate([cell_idx_a, cell_idx_b])
    print(adata.shape)
    adata = adata[cell_idx_all].copy()
    print(adata.shape)
    total_cells_per_gene = (adata.X > 0).sum(axis=0).A1.ravel()
    adata = adata[:, total_cells_per_gene >= min_total_cells_per_gene]
    print(adata.shape)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X_a = adata.X[: len(cell_idx_a)].toarray()
    X_b = adata.X[len(cell_idx_a) :].toarray()
    print(X_a.shape, X_b.shape)
    diff_exp = pd.DataFrame(
        dict(
            gene=adata.var_names.values,
            p_value=[
                mannwhitneyu(X_a[:, i], X_b[:, i], alternative="two-sided").pvalue
                for i in range(X_a.shape[1])
            ],
            lfc=np.log2(np.expm1(X_a).mean(axis=0) + 1e-9)
            - np.log2(np.expm1(X_b).mean(axis=0) + 1e-9),
        )
    )
    diff_exp["abs_lfc"] = diff_exp.lfc.abs()
    diff_exp["sort_val"] = list(zip(diff_exp.p_value, -diff_exp.lfc.abs()))
    diff_exp = diff_exp.sort_values(by="sort_val").drop("sort_val", 1)
    diff_exp["ranking"] = np.arange(len(diff_exp))
    return diff_exp


def run_differential_expression(
    adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene=100
):
    print("sample sizes: ", len(cell_idx_a), len(cell_idx_b))
    diff_exp = _run_differential_expression(
        adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene
    )
    reject, pvals_corrected, _, _ = multipletests(
        diff_exp.p_value.values, 0.05, "fdr_bh"
    )
    diff_exp["p_value_adj"] = pvals_corrected
    return diff_exp
