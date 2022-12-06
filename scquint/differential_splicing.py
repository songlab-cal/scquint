import anndata
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from pyro.distributions import Dirichlet, DirichletMultinomial, Gamma, Multinomial
import scanpy as sc
from scipy.special import softmax
from scipy.stats import chi2, mannwhitneyu
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .data import make_intron_group_summation_cpu, filter_min_cells_per_feature, filter_min_cells_per_intron_group, regroup, filter_min_global_proportion


# original: from skbio.stats.composition import closure
def closure(mat):
    """
    Performs closure to ensure that all elements add up to 1.
    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.
    ValueError
       Raises an error if there is a row that has all zeros.
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])
    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


# original function: from skbio.stats.composition import alr
def alr(mat, denominator_idx=0):
    r"""
    Performs additive log ratio transformation.
    This function transforms compositions from a D-part Aitchison simplex to
    a non-isometric real space of D-1 dimensions. The argument
    `denominator_col` defines the index of the column used as the common
    denominator. The :math: `alr` transformed data are amenable to multivariate
    analysis as long as statistics don't involve distances.
    :math:`alr: S^D \rightarrow \mathbb{R}^{D-1}`
    The alr transformation is defined as follows
    .. math::
        alr(x) = \left[ \ln \frac{x_1}{x_D}, \ldots,
        \ln \frac{x_{D-1}}{x_D} \right]
    where :math:`D` is the index of the part used as common denominator.
    Parameters
    ----------
    mat: numpy.ndarray
       a matrix of proportions where
       rows = compositions and
       columns = components
    denominator_idx: int
       the index of the column (2D-matrix) or position (vector) of
       `mat` which should be used as the reference composition. By default
       `denominator_idx=0` to specify the first column or position.
    Returns
    -------
    numpy.ndarray
         alr-transformed data projected in a non-isometric real space
         of D-1 dimensions for a D-parts composition
    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import alr
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr(x)
    array([ 1.09861229,  1.38629436,  0.69314718])
    """
    mat = closure(mat)
    if mat.ndim == 2:
        mat_t = mat.T
        numerator_idx = list(range(0, mat_t.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat_t[numerator_idx, :]/mat_t[denominator_idx, :]).T
    elif mat.ndim == 1:
        numerator_idx = list(range(0, mat.shape[0]))
        del numerator_idx[denominator_idx]
        lr = np.log(mat[numerator_idx]/mat[denominator_idx])
    else:
        raise ValueError("mat must be either 1D or 2D")
    return lr



def lrtest(llmin, llmax, df):
    lr = 2 * (llmax - llmin)
    p = chi2.sf(lr, df)
    return p


def normalize(x):
    return x / sum(x)


def run_regression(args):
    intron_group, y, cell_idx_a, cell_idx_b = args
    cells_to_use = np.where(y.sum(axis=1) > 0)[0]
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
        raise Exception(f"WARNING: optimization failed for intron_group {intron_group}. ll_null={ll_null} ll_full={ll}")
    p_value = lrtest(ll_null, ll, n_classes - 1)
    A = model.get_full_A().cpu().detach().numpy()
    log_alpha = model.log_alpha.cpu().detach().numpy()

    conc = np.exp(log_alpha)
    beta = A.T
    psi1 = normalize(conc * softmax(beta[:, 0]))
    psi2 = normalize(conc * softmax(beta.sum(axis=1)))
    if np.isnan(p_value): p_value = 1.0

    df_intron_group = pd.DataFrame(dict(intron_group=[intron_group], p_value=[p_value], ll_null=[ll_null], ll=[ll], n_classes=[n_classes]))
    df_intron = pd.DataFrame(dict(psi_a=psi1, psi_b=psi2))

    return df_intron_group, df_intron


def _run_differential_splicing(
    adata,
    cell_idx_a,
    cell_idx_b,
    device="cpu",
    min_cells_per_intron_group=None,
    min_total_cells_per_intron=None,
    n_jobs=None,
    do_regroup=False,
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
    if do_regroup:
        adata = regroup(adata)
        print(adata.shape)
    if min_cells_per_intron_group is not None:
        adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, cell_idx_a)
        adata = filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, cell_idx_b)
        print(adata.shape)
    if adata.shape[1] == 0: return pd.DataFrame(), pd.DataFrame()

    print("Number of intron groups: ", len(adata.var.intron_group.unique()))
    print("Number of introns: ", len(adata.var))

    intron_groups = adata.var.intron_group.values
    all_intron_groups = pd.unique(intron_groups)
    intron_group_introns = defaultdict(list)
    for i, c in enumerate(intron_groups):
        intron_group_introns[c].append(i)

    X = adata.X.toarray()  # for easier parallelization using Python's libraries

    if n_jobs is not None and n_jobs != 1:
        dfs_intron_group, dfs_intron = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(run_regression)((c, X[:, intron_group_introns[c]], cell_idx_a, cell_idx_b))
                for c in tqdm(all_intron_groups)
            )
        )
    else:
        dfs_intron_group, dfs_intron = zip(*[
            run_regression((c, X[:, intron_group_introns[c]], cell_idx_a, cell_idx_b))
            for c in tqdm(all_intron_groups)
        ])
    df_intron_group = pd.concat(dfs_intron_group, ignore_index=True)
    df_intron = pd.concat(dfs_intron, ignore_index=True)
    positions = np.concatenate([intron_group_introns[c] for c in all_intron_groups])
    df_intron = pd.concat([adata.var.iloc[positions].reset_index(drop=False), df_intron], axis=1).set_index("index")
    return df_intron_group, df_intron


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
        ll = DirichletMultinomial(concentration, validate_args=False).log_prob(Y).sum()
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

    df_intron_group, df_intron = _run_differential_splicing(
        adata,
        cell_idx_a,
        cell_idx_b,
        **kwargs,
    )
    if len(df_intron_group) == 0: return df_intron_group, df_intron
    df_intron["delta_psi"] = df_intron["psi_a"] - df_intron["psi_b"]
    df_intron["lfc_psi"] = np.log2(df_intron["psi_a"] + 1e-9) - np.log2(df_intron["psi_b"] + 1e-9)
    df_intron["abs_delta_psi"] = df_intron.delta_psi.abs()
    df_intron["abs_lfc_psi"] = df_intron.lfc_psi.abs()

    if "gene_id" not in df_intron.columns:
        df_intron["gene_id"] = "NA"
    if "gene_name" not in df_intron.columns:
        df_intron["gene_name"] = "NA"
    groupby = df_intron.groupby("intron_group").agg({"gene_id": "first", "gene_name": "first", "abs_delta_psi": "max", "abs_lfc_psi": "max"})
    groupby = groupby.rename(columns={"abs_delta_psi": "max_abs_delta_psi", "abs_lfc_psi": "max_abs_lfc_psi"})
    df_intron_group = df_intron_group.set_index("intron_group").merge(groupby, left_index=True, right_index=True, how="inner")
    df_intron_group = df_intron_group.sort_values(by="p_value")
    df_intron_group["ranking"] = np.arange(len(df_intron_group))

    reject, pvals_corrected, _, _ = multipletests(
        df_intron_group.p_value.values, 0.05, "fdr_bh"
    )
    df_intron_group["p_value_adj"] = pvals_corrected

    return df_intron_group, df_intron


def make_intron_group_summation(intron_groups, device):
    n_introns = len(intron_groups)
    n_intron_groups = len(np.unique(intron_groups))
    rows, cols = zip(*list(enumerate(intron_groups)))
    vals = np.ones(n_introns, dtype=int)

    I = torch.tensor([cols, rows], device=device, dtype=torch.long)
    V = torch.tensor(vals, device=device, dtype=torch.float)
    intron_group_summation = torch.sparse.FloatTensor(
        I, V, torch.Size([n_intron_groups, n_introns])
    )
    return intron_group_summation


class LassoDirichletMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, l1_penalty, init_A=None, init_log_alpha=None):
        super(LassoDirichletMultinomialGLM, self).__init__()
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
            + self.l1_penalty * torch.sum(torch.abs(self.A[1:]))  # excluding the intercept
        )
        self.ll = ll
        return res


class LassoMultinomialGLM(nn.Module):
    def __init__(self, n_covariates, n_classes, l1_penalty):
        super(LassoMultinomialGLM, self).__init__()
        self.A = nn.Parameter(torch.zeros((n_covariates, n_classes-1), dtype=torch.double))
        self.register_buffer("constant_column", torch.zeros((n_covariates, 1), dtype=torch.double))
        self.ll = None
        self.l1_penalty = l1_penalty

    def get_full_A(self):
        return torch.cat([self.A, self.constant_column], 1)

    def forward(self, X):
        A = self.get_full_A()
        logits = X @ A
        return logits

    def loss_function(self, X, Y):
        logits = self.forward(X)
        ll = Multinomial(logits=logits).log_prob(Y).sum()
        res = (
            - ll
            + self.l1_penalty * torch.sum(torch.abs(self.A[1:]))  # excluding the intercept
        )
        self.ll = ll
        return res


class CVLassoMultinomialGLM:
    def __init__(self, l1_penalty_min, l1_penalty_max, n_trials):
        self.l1_penalty_min = l1_penalty_min
        self.l1_penalty_max = l1_penalty_max
        self.n_trials = n_trials

        import warnings
        warnings.filterwarnings('ignore')

    def fit(self, X, Y, stratification, device="cpu"):
        n_covariates = X.shape[1]
        n_classes = Y.shape[1]

        def objective(l1_penalty):
            train_ll = []
            test_ll = []
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, stratification):
                model = lambda: LassoMultinomialGLM(n_covariates, n_classes, l1_penalty)
                ll, model = fit_model(model, X[train_index], Y[train_index], device=device)
                train_ll.append(ll)
                model.loss_function(
                    torch.tensor(X[test_index], dtype=torch.double, device=device),
                    torch.tensor(Y[test_index], dtype=torch.double, device=device)
                )
                test_ll.append(model.ll.cpu().detach().numpy())
            return -np.mean(test_ll)

        l1_penalties = np.linspace(self.l1_penalty_min, self.l1_penalty_max, self.n_trials)
        losses = list(map(objective, l1_penalties))
        l1_penalty = l1_penalties[np.argmin(losses)]

        self.l1_penalty = l1_penalty
        model = lambda: LassoMultinomialGLM(n_covariates, n_classes, l1_penalty)
        ll, model = fit_model(model, X, Y, device=device)
        self.ll = ll
        self.model = model


class CVLassoDirichletMultinomialGLM:
    def __init__(self, l1_penalty_min, l1_penalty_max, n_trials):
        self.l1_penalty_min = l1_penalty_min
        self.l1_penalty_max = l1_penalty_max
        self.n_trials = n_trials

        import warnings
        warnings.filterwarnings('ignore')

    def fit(self, X, Y, stratification, device="cpu", threads=1):
        n_covariates = X.shape[1]
        n_classes = Y.shape[1]

        def objective(l1_penalty, X, Y, stratification):
            train_ll = []
            test_ll = []
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, stratification):
                model = lambda: LassoDirichletMultinomialGLM(n_covariates, n_classes, l1_penalty)
                ll, model = fit_model(model, X[train_index], Y[train_index], device=device)
                train_ll.append(ll)
                model.loss_function(
                    torch.tensor(X[test_index], dtype=torch.double, device=device),
                    torch.tensor(Y[test_index], dtype=torch.double, device=device)
                )
                test_ll.append(model.ll.cpu().detach().numpy())
            return -np.mean(test_ll)

        l1_penalties = np.linspace(self.l1_penalty_min, self.l1_penalty_max, self.n_trials)
        losses =  Parallel(n_jobs=threads)(delayed(objective)(l1_penalty, X, Y, stratification) for l1_penalty in l1_penalties)
        print("losses: ", losses)
        l1_penalty = l1_penalties[np.nanargmin(losses)]

        self.l1_penalty = l1_penalty
        model = lambda: LassoDirichletMultinomialGLM(n_covariates, n_classes, l1_penalty)
        ll, model = fit_model(model, X, Y, device=device)
        self.ll = ll
        self.model = model


def _run_differential_expression(adata, cell_idx_a, cell_idx_b, min_total_cells_per_gene):
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


def run_differential_splicing_for_each_group(
    adata_spl,
    groupby,
    groups=None,
    subset_to_groups=False,
    **kwargs,
):
    if subset_to_groups:
        assert(groups is not None)
        adata_spl = adata_spl[adata_spl.obs[groupby].isin(groups)]
    
    all_intron_groups = []
    all_introns = []

    if groups is None:
        groups = adata_spl.obs[groupby].unique()

    for g in groups:
        print(g)
        cell_idx_a = np.where(adata_spl.obs[groupby]==g)[0]
        cell_idx_b = np.where(adata_spl.obs[groupby]!=g)[0]
        intron_groups, introns = run_differential_splicing(
            adata_spl, cell_idx_a, cell_idx_b, **kwargs, 
        )
        intron_groups["test_group"] = g
        introns["test_group"] = g
        intron_groups["name"] = intron_groups.index
        introns["name"] = introns.index
        all_intron_groups.append(intron_groups)
        all_introns.append(introns)

    all_intron_groups = pd.concat(all_intron_groups, ignore_index=True)
    all_introns = pd.concat(all_introns, ignore_index=True)
    return all_intron_groups, all_introns


def find_marker_introns(intron_groups, introns, n=10, max_p_value_adj=0.05, min_delta_psi=0.05):
    intron_groups = intron_groups[intron_groups.p_value_adj <= max_p_value_adj]
    significant_groups_per_ig = intron_groups.groupby(["name"]).test_group.unique()
    groups = intron_groups.test_group.unique()
    
    def check_sig(i):
        return (
            i.intron_group in significant_groups_per_ig.index and
            i.test_group in significant_groups_per_ig.loc[i.intron_group] and
            i.delta_psi >= min_delta_psi
        )
    introns = introns[introns.apply(check_sig, axis=1)]

    marker_introns = defaultdict(list)
    introns = introns.sample(frac=1.0, random_state=42).sort_values("delta_psi", ascending=False).drop_duplicates("gene_name")
    i = 0
    while sum(map(len, marker_introns.values())) < n*len(groups) and i < len(introns):
        intron = introns.iloc[i]
        if len(marker_introns[intron.test_group]) < n:
            marker_introns[intron.test_group].append(intron["name"])
        i += 1
    return marker_introns


def mask_PSI(adata_spl, marker_introns, groupby, min_cells=10):
    marker_introns_list = sum(marker_introns.values(), [])
    adata_spl = adata_spl[:, marker_introns_list]
    PSI_raw = pd.DataFrame(adata_spl.layers["PSI_raw"])
    n_cells_per_group = PSI_raw.notna().groupby(adata_spl.obs[groupby].values).sum()
    PSI_raw_masked = adata_spl.layers["PSI_raw"].copy()
    for g in n_cells_per_group.index:
        idx_cells = np.where(adata_spl.obs[groupby]==g)[0]
        for i in n_cells_per_group.columns:
            if n_cells_per_group.loc[g, i] < min_cells:
                PSI_raw_masked[idx_cells, i] = np.nan
    adata_spl.layers["PSI_raw_masked"] = PSI_raw_masked
    return adata_spl
