import anndata
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from skbio.stats.composition import alr
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from pyro.distributions import Dirichlet, DirichletMultinomial, Gamma, Multinomial
from scipy.special import softmax
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from .utils import make_cluster_summation_cpu, relabel, group_normalize, filter_min_cells_per_feature, filter_min_cells_per_cluster, recluster, filter_min_global_proportion


def lrtest(llmin, llmax, df):
    lr = 2 * (llmax - llmin)
    p = chi2.sf(lr, df)
    return p


def normalize(x):
    return x / sum(x)


def run_regression(args):
    cluster, y, cell_idx_a, cell_idx_b = args
    #var = adata.var.copy()
    #cluster = var.iloc[0].cluster
    if cluster % 100 == 0:
        print("Testing intron cluster ", cluster)
    #y = adata.X
    #cells_to_use = np.where(y.sum(axis=1).A1 > 0)[0]
    cells_to_use = np.where(y.sum(axis=1) > 0)[0]
    #y = y[cells_to_use].toarray()
    y = y[cells_to_use]
    n_cells, n_classes = y.shape
    n_covariates = 2
    cell_mask_a = np.isin(cells_to_use, cell_idx_a)
    cell_mask_b = np.isin(cells_to_use, cell_idx_b)
    x = np.ones((n_cells, 2), dtype=float)
    x[cell_mask_a, 1] = 0
    x_null = np.expand_dims(x[:, 0], axis=1)

    pseudocounts = 10.0
    init_A_null = np.expand_dims(alr(y.sum(axis=0) + pseudocounts, denominator_idx=-1), axis=0)
    model_null = lambda: DirichletMultinomialGLM(1, n_classes, init_A=init_A_null)

    ll_null, model_null = fit_model(model_null, x_null, y)
    init_A = np.zeros((2, n_classes - 1), dtype=float)
    init_A[0] = alr(y[cell_mask_a].sum(axis=0) + pseudocounts, denominator_idx=-1)
    init_A[1] = alr(y[cell_mask_b].sum(axis=0) + pseudocounts, denominator_idx=-1) - init_A[0]
    model = lambda: DirichletMultinomialGLM(2, n_classes, init_A=init_A)
    ll, model = fit_model(model, x, y)
    if ll+1e-2 < ll_null:
        raise Exception(f"WARNING: optimization failed for cluster {cluster}. ll_null={ll_null} ll_full={ll}")
    p_value = lrtest(ll_null, ll, n_classes - 1)
    A = model.get_full_A().cpu().detach().numpy()
    log_alpha = model.log_alpha.cpu().detach().numpy()

    conc = np.exp(log_alpha)
    beta = A.T
    psi1 = normalize(conc * softmax(beta[:, 0]))
    psi2 = normalize(conc * softmax(beta.sum(axis=1)))
    if np.isnan(p_value): p_value = 1.0

    df_cluster = pd.DataFrame(dict(cluster=[cluster], p_value=[p_value], ll_null=[ll_null], ll=[ll], n_classes=[n_classes]))
    df_intron = pd.DataFrame(dict(psi_a=psi1, psi_b=psi2))

    return df_cluster, df_intron


def _run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    device="cpu",
    min_cells_per_cluster=None,
    min_total_cells_per_intron=None,
    n_jobs=None,
    do_recluster=False,
    min_global_proportion=1e-3,
):
    n_a = len(cell_idx_a)
    n_b = len(cell_idx_b)
    cell_idx_all = np.concatenate([cell_idx_a, cell_idx_b])
    adata = adata[cell_idx_all].copy()
    cell_idx_a = np.arange(0, n_a)
    cell_idx_b = np.arange(n_a, n_a + n_b)
    print(adata.shape)
    if min_total_cells_per_intron is not None:
        adata = filter_min_cells_per_feature(adata, min_total_cells_per_intron)
        print(adata.shape)
    if min_global_proportion is not None:
        adata = filter_min_global_proportion(adata, min_global_proportion)
        print(adata.shape)
    if do_recluster:
        adata = recluster(adata)
        print(adata.shape)
    if min_cells_per_cluster is not None:
        adata = filter_min_cells_per_cluster(adata, min_cells_per_cluster, cell_idx_a)
        adata = filter_min_cells_per_cluster(adata, min_cells_per_cluster, cell_idx_b)
        print(adata.shape)
    if adata.shape[1] == 0: return pd.DataFrame(), pd.DataFrame()

    adata.var = adata.var.reset_index(drop=True)
    print("Number of intron clusters: ", len(adata.var.cluster.unique()))
    print("Number of introns: ", len(adata.var))

    intron_clusters = adata.var.cluster.values
    all_intron_clusters = pd.unique(intron_clusters)
    cluster_introns = defaultdict(list)
    for i, c in enumerate(intron_clusters):
        cluster_introns[c].append(i)

    #print("subsetting to 2000")
    #all_intron_clusters = all_intron_clusters[:1000]

    X = adata.X.toarray()  # for easier parallelization using Python's libraries

    if n_jobs is not None and n_jobs > 1:
        dfs_cluster, dfs_intron = zip(
            *Parallel(n_jobs=n_jobs)(
            #*Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(run_regression)((c, X[:, cluster_introns[c]], cell_idx_a, cell_idx_b))
                for c in all_intron_clusters
            )
        )

        #pool = multiprocessing.Pool(n_jobs)
        #tested_clusters, tested_p_values, ll_null, ll, n_classes, tested_vars = zip(
        #    #*pool.map(
        #    *pool.imap_unordered(
        #        run_regression,
        #        #((adata[:, cluster_introns[c]], cell_idx_a, cell_idx_b) for c in all_intron_clusters),
        #        ((c, X[:, cluster_introns[c]], cell_idx_a, cell_idx_b) for c in all_intron_clusters),
        #        32,
        #        #((1, 2, 3) for c in all_intron_clusters),
        #    )
        #)
    else:
        tested_clusters, tested_p_values, ll_null, ll, n_classes, tested_vars = zip(*[run_regression(i) for i in adata.var.cluster.unique()])
    df_cluster = pd.concat(dfs_cluster, ignore_index=True)
    df_intron = pd.concat(dfs_intron, ignore_index=True)
    positions = np.concatenate([cluster_introns[c] for c in all_intron_clusters])
    df_intron = pd.concat([adata.var.iloc[positions].reset_index(drop=True), df_intron], axis=1)
    print("Done")
    return df_cluster, df_intron


class MultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes):
        super(MultinomialGLM, self).__init__()
        self.A = nn.Parameter(torch.zeros((n_covariates, n_classes-1), dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        A = self.get_full_A()
        logits = X @ A
        return logits

    def loss_function(self, X, Y):
        logits = self.forward(X)
        ll = Multinomial(logits=logits).log_prob(Y).sum()
        self.ll = ll
        if torch.isnan(ll):
            print("A: ", self.A)
            print("ll: ", ll)
            raise Exception("debug")
        return -ll


class DirichletMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, init_A=None, init_log_alpha=None):
        super(DirichletMultinomialGLM, self).__init__()
        self.n_covariates = n_covariates
        self.n_classes = n_classes
        if init_A is None:
            init_A = np.zeros((n_covariates, n_classes - 1))
        if init_log_alpha is None:
            init_log_alpha = np.ones(1) * 1.0
        self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.double))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.register_buffer("conc_shape", torch.tensor(1 + 1e-4, dtype=torch.double))
        self.register_buffer("conc_rate", torch.tensor(1e-4, dtype=torch.double))
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        alpha = torch.exp(self.log_alpha)
        A = self.get_full_A()
        P = torch.softmax(X @ A, dim=1)
        concentration = torch.mul(alpha, P)
        return A, alpha, concentration, P

    def loss_function(self, X, Y):
        A, alpha, concentration, P = self.forward(X)
        ll = DirichletMultinomial(concentration).log_prob(Y).sum()
        res = (
            - ll
            - Gamma(self.conc_shape, self.conc_rate).log_prob(alpha).sum()
        )
        self.ll = ll
        return res


def fit_model(model_initializer, X, Y, device="cpu"):
    X = torch.tensor(X, dtype=torch.double, device=device)
    Y = torch.tensor(Y, dtype=torch.double, device=device)

    initial_lr = 1.0

    def try_optimization(lr):
        model = model_initializer()
        model.to(device)
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=10000, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = model.loss_function(X, Y)
            if torch.isnan(loss):
                raise ValueError("nan encountered")
            loss.backward()
            return loss

        optimizer.step(closure)
        return model.ll.cpu().detach().numpy(), model

    lr = initial_lr
    try_number = 0
    while True:
        try_number += 1
        if try_number > 10:
            print("WARNING: optimization failed, too many tries")
            return -np.inf, model_initializer()
        try:
            ll, model = try_optimization(lr)
            break
        except ValueError as ve:
            lr /= 10.0

    return ll, model


def run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    **kwargs
):
    print("sample sizes: ", len(cell_idx_a), len(cell_idx_b))

    df_cluster, df_intron = _run_differential_splicing(
        adata,
        cell_idx_a,
        cell_idx_b,
        **kwargs,
    )
    if len(df_cluster) == 0: return df_cluster, df_intron
    df_intron["delta_psi"] = df_intron["psi_a"] - df_intron["psi_b"]
    df_intron["lfc_psi"] = np.log2(df_intron["psi_a"] + 1e-9) - np.log2(df_intron["psi_b"] + 1e-9)
    df_intron["abs_delta_psi"] = df_intron.delta_psi.abs()
    df_intron["abs_lfc_psi"] = df_intron.lfc_psi.abs()

    groupby = df_intron.groupby("cluster").agg({"gene_id": "first", "abs_delta_psi": "max", "abs_lfc_psi": "max"})
    groupby = groupby.rename(columns={"abs_delta_psi": "max_abs_delta_psi", "abs_lfc_psi": "max_abs_lfc_psi"})
    df_cluster = df_cluster.set_index("cluster").merge(groupby, left_index=True, right_index=True, how="inner")
    df_cluster = df_cluster.sort_values(by="p_value")
    df_cluster["ranking"] = np.arange(len(df_cluster))

    reject, pvals_corrected, _, _ = multipletests(
        df_cluster.p_value.values, 0.05, "fdr_bh"
    )
    df_cluster["p_value_adj"] = pvals_corrected

    return df_cluster, df_intron


def make_cluster_summation(clusters, device):
    n_introns = len(clusters)
    n_clusters = len(np.unique(clusters))
    rows, cols = zip(*list(enumerate(clusters)))
    vals = np.ones(n_introns, dtype=int)

    I = torch.tensor([cols, rows], device=device, dtype=torch.long)
    V = torch.tensor(vals, device=device, dtype=torch.float)
    cluster_summation = torch.sparse.FloatTensor(
        I, V, torch.Size([n_clusters, n_introns])
    )
    return cluster_summation


class RegularizedDirichletMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, l1_penalty=0.0, init_A=None, init_log_alpha=None):
        super(RegularizedDirichletMultinomialGLM, self).__init__()
        self.n_covariates = n_covariates
        self.n_classes = n_classes
        if init_A is None:
            init_A = np.zeros((n_covariates, n_classes - 1))
        if init_log_alpha is None:
            init_log_alpha = np.ones(1) * 1.0
        self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.double))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.register_buffer("conc_shape", torch.tensor(1 + 1e-4, dtype=torch.double))
        self.register_buffer("conc_rate", torch.tensor(1e-4, dtype=torch.double))
        self.l1_penalty = l1_penalty
        self.ll = None

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        alpha = torch.exp(self.log_alpha)
        A = self.get_full_A()
        P = torch.softmax(X @ A, dim=1)
        concentration = torch.mul(alpha, P)
        return A, alpha, concentration, P

    def loss_function(self, X, Y):
        A, alpha, concentration, P = self.forward(X)
        ll = DirichletMultinomial(concentration).log_prob(Y).sum()
        res = (
            - ll
            - Gamma(self.conc_shape, self.conc_rate).log_prob(alpha).sum()
            + self.l1_penalty * torch.sum(torch.abs(self.A))
        )
        self.ll = ll
        return res
