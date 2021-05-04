import anndata
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

from .utils import make_cluster_summation_cpu, relabel, group_normalize, filter_min_cells_per_feature, filter_min_cells_per_cluster


def lrtest(llmin, llmax, df):
    lr = 2 * (llmax - llmin)
    p = chi2.sf(lr, df)
    return p


def normalize(x):
    return x / sum(x)


def _run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    device="cpu",
    min_cells_per_cluster=None,
    min_total_cells_per_intron=None,
    n_jobs=None,
):
    n_a = len(cell_idx_a)
    n_b = len(cell_idx_b)
    cell_idx_all = np.concatenate([cell_idx_a, cell_idx_b])
    adata = adata[cell_idx_all].copy()
    #adata.var["original_cluster"] = adata.var.cluster
    cell_idx_a = np.arange(0, n_a)
    cell_idx_b = np.arange(n_a, n_a + n_b)
    if min_total_cells_per_intron is not None:
        adata = filter_min_cells_per_feature(adata, min_total_cells_per_intron)
    if min_cells_per_cluster is not None:
        adata = filter_min_cells_per_cluster(adata, min_cells_per_cluster, cell_idx_a)
        adata = filter_min_cells_per_cluster(adata, min_cells_per_cluster, cell_idx_b)
    if adata.shape[1] == 0: return pd.DataFrame(), pd.DataFrame()
    print("Number of intron clusters: ", len(adata.var.cluster.unique()))
    print("Number of introns: ", len(adata.var))


    def run_regression(i):
        if i % 100 == 0:
            print("Testing intron cluster ", i)
        adata_cluster = adata[:, adata.var.cluster==i].copy()  # Excessive copying. Else it's giving a lot of warnings
        cells_to_use = np.where(adata_cluster.X.sum(axis=1).A1 > 0)[0]
        adata_cluster = adata_cluster[cells_to_use].copy()
        y = adata_cluster.X.toarray()
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
        psi_all = normalize(y.sum(axis=0))
        psi_a = normalize(y[cell_mask_a].sum(axis=0))
        psi_b = normalize(y[cell_mask_b].sum(axis=0))
        ll_null, model_null = fit_model(model_null, x_null, y)
        init_A = np.zeros((2, n_classes - 1), dtype=float)
        init_A[0] = alr(y[cell_mask_a].sum(axis=0) + pseudocounts, denominator_idx=-1)
        init_A[1] = alr(y[cell_mask_b].sum(axis=0) + pseudocounts, denominator_idx=-1) - init_A[0]
        model = lambda: DirichletMultinomialGLM(2, n_classes, init_A=init_A)
        ll, model = fit_model(model, x, y)
        p_value = lrtest(ll_null, ll, n_classes - 1)
        A = model.get_full_A().cpu().detach().numpy()
        log_alpha = model.log_alpha.cpu().detach().numpy()

        conc = np.exp(log_alpha)
        beta = A.T
        psi1 = normalize(conc * softmax(beta[:, 0]))
        psi2 = normalize(conc * softmax(beta.sum(axis=1)))
        if np.isnan(p_value): p_value = 1.0

        adata_cluster.var["psi_a"] = psi1
        adata_cluster.var["psi_b"] = psi2

        cluster = adata_cluster.var.original_cluster.iloc[0]
        adata_cluster.var.loc[:, "cluster"] = cluster
        return cluster, p_value, ll_null, ll, n_classes, adata_cluster.var

    if n_jobs is not None and n_jobs > 1:
        tested_clusters, tested_p_values, ll_null, ll, n_classes, tested_vars = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(run_regression)(i)
                for i in adata.var.cluster.unique()
            )
        )
    else:
        tested_clusters, tested_p_values, ll_null, ll, n_classes, tested_vars = zip(*[run_regression(i) for i in adata.var.cluster.unique()])
    df_cluster = pd.DataFrame(dict(cluster=tested_clusters, p_value=tested_p_values, ll_null=ll_null, ll=ll, n_classes=n_classes))
    df_intron = pd.concat(tested_vars, ignore_index=True)
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
        self.register_buffer("P_regularization", torch.full((self.n_classes,), 1.005, dtype=torch.double))
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
            - Dirichlet(self.P_regularization).log_prob(P).sum()
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
