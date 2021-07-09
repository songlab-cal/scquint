import os
from collections import Counter
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp_sparse
from sklearn.decomposition import PCA
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyro.distributions import DirichletMultinomial, Gamma
from scipy import sparse as sp_sparse
from scvi.dataset.dataset import GeneExpressionDataset, compute_library_size
from scvi.inference import Posterior as scVIPosterior
from scvi.inference import UnsupervisedTrainer as UnsupervisedTrainer_scVI
from scvi.models.log_likelihood import log_nb_positive, log_zinb_positive
from scvi.models.modules import DecoderSCVI, Encoder, FCLayers
from scvi.models.utils import one_hot
from torch.distributions import Dirichlet, Laplace, Multinomial, Normal
from torch.distributions import kl_divergence as kl

from .utils import group_normalize


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


class Dataset(GeneExpressionDataset):
    def __init__(self, adata):
        self.n_genes = 0  # this is for pieces of code that rely on this
        self.genes = []
        self.obs = adata.obs
        self.var = adata.var
        self.n_cells = len(self.obs)
        print("n_cells: ", self.n_cells)
        self.n_introns = len(self.var)
        self.n_clusters = int(self.var.cluster.max() + 1)
        self.intron_clusters = self.var.cluster.values.astype(int)
        print("n_clusters: ", self.n_clusters)
        X = adata.X

        # batch_indices = None
        batch_indices = np.arange(
            self.n_cells
        )  # for non-amortized models that require it
        super(Dataset, self).__init__()
        self.populate_from_data(X, batch_indices=batch_indices)

    def filter_cells(self, idx):
        print(self.X.shape)
        self.X = self.X[idx]
        print(self.X.shape)
        self.obs = self.obs.iloc[idx]
        self.n_cells = len(self.obs)
        print(self.n_cells)

    def compute_library_size_batch(self):
        """Computes the library size per batch."""
        print("WARNING: omitting compute_library_size_batch")
        self.local_means = np.zeros((self.nb_cells, 1))
        self.local_vars = np.zeros((self.nb_cells, 1))
        self.cell_attribute_names.update(["local_means", "local_vars"])
        return

        self.local_means = np.zeros((self.nb_cells, 1))
        self.local_vars = np.zeros((self.nb_cells, 1))

        for i_batch in range(self.n_batches):
            idx_batch = np.squeeze(self.batch_indices == i_batch)

            (
                self.local_means[idx_batch],
                self.local_vars[idx_batch],
            ) = compute_library_size(self.X[idx_batch, :0])

        self.cell_attribute_names.update(["local_means", "local_vars"])


torch.backends.cudnn.benchmark = True


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def _log_beta_1(alpha, value, is_sparse):
    if is_sparse:
        mask = value != 0
        value, alpha, mask = torch.broadcast_tensors(value, alpha, mask)
        result = torch.zeros_like(value)
        value = value[mask]
        alpha = alpha[mask]
        result[mask] = (
            torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)
        )
        return result
    else:
        return (
            torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)
        )


class LinearEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super().__init__()
        self.mean_encoder = nn.Linear(n_input, n_output, bias=True)
        self.var_encoder = nn.Linear(n_input, n_output, bias=True)

    def forward(self, x: torch.Tensor, *cat_list: int):
        q_m = self.mean_encoder(x)
        q_v = torch.exp(self.var_encoder(x)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


class IntronsDecoder(nn.Module):
    def __init__(
        self,
        intron_clusters,
        cluster_summation,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.intron_clusters = intron_clusters
        self.cluster_summation = cluster_summation

        self.first_part = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.second_part = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor, first_indices):
        potentials = self.second_part(self.first_part(z))
        potentials[:, first_indices] = 0.0
        p_u = torch.exp(potentials)
        cluster_sums = torch.sparse.mm(self.cluster_summation, p_u.T).T
        norm_factor = cluster_sums[:, self.intron_clusters]
        p = torch.div(p_u, norm_factor)
        return p


class LinearIntronsDecoder(nn.Module):
    def __init__(
        self, intron_clusters, cluster_summation, n_input: int, n_output: int, bias=True
    ):
        super().__init__()
        self.intron_clusters = intron_clusters
        self.cluster_summation = cluster_summation

        self.first_part = nn.Linear(n_input, n_output, bias=bias)

    def forward(self, z: torch.Tensor, first_indices):
        potentials = self.first_part(z)
        potentials[:, first_indices] = 0.0
        p_u = torch.exp(potentials)
        cluster_sums = torch.sparse.mm(self.cluster_summation, p_u.T).T
        norm_factor = cluster_sums[:, self.intron_clusters]
        p = torch.div(p_u, norm_factor)
        return p


class VAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_introns: int,
        n_clusters: int,
        intron_clusters: [int],
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
        linearly_encoded=False,
        linearly_decoded=False,
        loss_genes="zinb",
        loss_introns="dirichlet-multinomial",
        weight_expression=1.0,
        weight_splicing=1.0,
        input_transform="log",
        feature_addition=None,
        regularization_gaussian_std=None,
        regularization_laplace_std=None,
        concentration_shape="cluster",
    ):
        super().__init__()
        self.log_variational = False
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_genes = n_genes
        self.n_introns = n_introns
        self.n_clusters = n_clusters
        self.use_cuda = use_cuda
        self.device = "cuda:0" if self.use_cuda else "cpu"
        print("self.device: ", self.device)
        self.loss_genes = loss_genes
        self.loss_introns = loss_introns
        self.weight_expression = weight_expression
        self.weight_splicing = weight_splicing
        self.input_transform = input_transform
        self.regularization_gaussian_std = regularization_gaussian_std
        self.regularization_laplace_std = regularization_laplace_std
        self.concentration_shape = concentration_shape

        all_intron_clusters = np.unique(intron_clusters)
        first_indices_dict = {}
        not_first_indices = []
        for i, c in enumerate(intron_clusters):
            if c not in first_indices_dict:
                first_indices_dict[c] = i
            else:
                not_first_indices.append(i)
        first_indices = np.array([first_indices_dict[c] for c in all_intron_clusters])
        not_first_indices = np.array(not_first_indices)
        print(len(intron_clusters), len(first_indices), len(not_first_indices), len(first_indices)+len(not_first_indices))

        n_cat_list = [n_batch]

        if self.n_genes > 0:
            self.px_r = torch.nn.Parameter(
                torch.randn(
                    n_genes,
                )
            )
            self.l_encoder = Encoder(
                n_genes, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
            )
            self.genes_decoder = DecoderSCVI(
                n_latent,
                n_genes,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
            )

        input_dimension = n_genes + n_introns - n_clusters if input_transform == "frequency-smoothed" else n_genes + n_introns
        if linearly_encoded:
            self.z_encoder = LinearEncoder(input_dimension, n_latent)
        else:
            self.z_encoder = Encoder(
                input_dimension,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )

        if self.n_introns > 0:
            self.first_indices = torch.tensor(
                first_indices, dtype=torch.long, device=self.device
            )
            self.not_first_indices = torch.tensor(
                not_first_indices, dtype=torch.long, device=self.device
            )

            self.intron_clusters = torch.tensor(
                intron_clusters, dtype=torch.long, device=self.device
            )
            cols, rows = zip(*list(enumerate(intron_clusters)))
            vals = np.ones(len(intron_clusters), dtype=int)

            I = torch.tensor([rows, cols], device=self.device, dtype=torch.long)
            V = torch.tensor(vals, device=self.device, dtype=torch.float)
            self.cluster_summation = torch.sparse.FloatTensor(
                I, V, torch.Size([self.n_clusters, self.n_introns])
            )

            # intron_sums = np.ones_like(intron_sums)
            # self.intron_sums = torch.tensor(intron_sums, dtype=torch.float, device=self.device)

            if input_transform == "frequency-smoothed":
                assert len(feature_addition) == n_introns
                self.feature_addition = torch.tensor(
                    feature_addition, device=self.device, dtype=torch.float
                )

            if self.use_cuda:
                self.cluster_summation = self.cluster_summation.cuda()

            # cluster_sums = (torch.sparse.mm(self.cluster_summation, self.intron_sums.reshape(n_introns, 1))).reshape(n_clusters)[self.intron_clusters]

            # self.mean_factors = torch.log(self.intron_sums+cluster_sums) - torch.log(cluster_sums)

            # self.mean_factors = [np.log(1+1.0/(intron_clusters==c).sum()) for c in intron_clusters]
            # self.mean_factors = torch.tensor(self.mean_factors, dtype=torch.float)

            # print(self.mean_factors[:3])
            # raise Exception('debug')

            # foo = [1.0/(intron_clusters==c).sum() for c in intron_clusters]
            # self.foo = torch.tensor(foo, dtype=torch.float).cuda()

            if linearly_decoded:
                self.introns_decoder = LinearIntronsDecoder(
                    self.intron_clusters,
                    self.cluster_summation,
                    n_latent,
                    n_introns,
                    bias=True,
                )
            else:
                self.introns_decoder = IntronsDecoder(
                    self.intron_clusters,
                    self.cluster_summation,
                    n_input=n_latent,
                    n_output=n_introns,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    n_cat_list=None,
                )

            if loss_introns == "dirichlet-multinomial":
                if self.concentration_shape == "intron":
                    self.feature_precision = torch.nn.Parameter(torch.randn(n_introns))
                elif self.concentration_shape == "cluster":
                    self.feature_precision = torch.nn.Parameter(torch.randn(n_clusters))

    def reconstruction_loss_genes(self, x, px_rate, px_r, px_dropout):
        if self.loss_genes == "zinb":
            return -log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)
        elif self.loss_genes == "nb":
            return -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
        else:
            raise Exception("reconstruction loss not implemented")

    def reconstruction_loss_introns(self, x, p):
        if self.loss_introns == "multinomial":
            return -self.multinomial_log_likelihood(x, p)
        elif self.loss_introns == "dirichlet-multinomial":
            if self.concentration_shape == "intron":
                return -self.dirichlet_multinomial_log_likelihood(
                    x, p, torch.exp(self.feature_precision)
                )
            elif self.concentration_shape == "cluster":
                return -self.dirichlet_multinomial_log_likelihood(
                    x, p, torch.exp(self.feature_precision)[self.intron_clusters]
                )
        else:
            raise Exception("reconstruction loss not implemented")

    def get_latents(self, x, y=None):
        r"""returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r"""samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        qz_m, qz_v, z = self.z_encoder(
            self.transform_input(x), y
        )  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        r"""samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[0]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[2]

    @torch.no_grad()
    def get_sample_alpha_posterior(self, posterior, n_samples):
        alphas = []
        for tensors in posterior:
            sample_batch, _, _, batch_index, labels = tensors
            alphas.append(
                self.get_sample_alpha(
                    sample_batch, batch_index=batch_index, y=labels, n_samples=n_samples
                )
            )
        return torch.squeeze(torch.cat(alphas))

    @torch.no_grad()
    def get_sample_alpha(self, x, batch_index=None, y=None, n_samples=1):
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[10]

    @torch.no_grad()
    def get_factor_loadings(self, center_around_zero=False):
        loadings = self.introns_decoder.first_part.weight.clone()
        loadings[self.first_indices] = 0.0
        print(loadings.shape)
        if center_around_zero:
            cluster_counter = collections.Counter(
                [int(x) for x in self.intron_clusters]
            )
            cluster_sizes = torch.tensor(
                [cluster_counter[c] for c in range(self.n_clusters)],
                dtype=torch.float,
                device=self.device,
            )
            print(cluster_sizes.shape)
            print(cluster_sizes[:10])
            # intron_cluster_sizes = cluster_sizes[self.intron_clusters]
            cluster_sums = torch.sparse.mm(self.cluster_summation, loadings)
            print(cluster_sums.shape)
            print(cluster_sums.T[0, :10])
            cluster_means = torch.div(cluster_sums.T, cluster_sizes).T
            print(cluster_means.shape)
            intron_cluster_means = cluster_means[self.intron_clusters]
            print(intron_cluster_means.shape)
            loadings -= intron_cluster_means
        return loadings.T.cpu().detach().numpy()

    # def scale_from_z(self, sample_batch, fixed_batch):
    #     qz_m, qz_v, z = self.z_encoder(sample_batch)
    #     batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
    #     library = 4. * torch.ones_like(sample_batch[:, [0]])
    #     px_scale, _, _, _ = self.genes_decoder('gene', z, library, batch_index)
    #     return px_scale

    def multinomial_log_likelihood(self, x, p):
        # part that depends on alpha
        log_p = torch.log(p)
        log_powers = (x * log_p).sum(dim=1)
        # print("log_powers.shape", log_powers.shape)

        log_factorial_xs = torch.lgamma(x + 1).sum(-1)
        # print("log_factorial_xs.shape", log_factorial_xs.shape)

        x_cluster_sums = torch.sparse.mm(self.cluster_summation, x.T).T
        # print("x_cluster_sums.shape", x_cluster_sums.shape)
        log_factorial_n = torch.lgamma(x_cluster_sums + 1).sum(-1)
        # print("log_factorial_n.shape", log_factorial_n.shape)

        return log_factorial_n - log_factorial_xs + log_powers

    def dirichlet_multinomial_log_likelihood(self, x, p, feature_precision):
        # print("p.shape", p.shape)
        # print("feature_precision.shape", feature_precision.shape)
        alpha = p * feature_precision
        # print("alpha.shape", alpha.shape)

        x_sum = torch.sparse.mm(self.cluster_summation, x.T).T
        alpha_sum = torch.sparse.mm(self.cluster_summation, alpha.T).T

        t1 = _log_beta_1(alpha_sum, x_sum, True).sum(dim=1)
        t2 = _log_beta_1(alpha, x, True).sum(dim=1)
        res = t1 - t2
        # print("res.shape", res.shape)
        return res

    def transform_input(self, x):
        if self.input_transform == "log":
            return torch.log(1 + x)
        elif self.input_transform == "frequency-smoothed":
            x2 = x + self.feature_addition
            cluster_sums = torch.sparse.mm(self.cluster_summation, x2.T).T
            norm_factor = cluster_sums[:, self.intron_clusters]
            x2 = torch.div(x2, norm_factor)
            return x2[:, self.not_first_indices]
        else:
            raise Exception("input_transform not implemented", self.input_transform)

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x

        px_scale = (
            px_r
        ) = (
            px_rate
        ) = px_dropout = qz_m = qz_v = z = ql_m = ql_v = library = alpha = None

        x2 = self.transform_input(x)
        qz_m, qz_v, z = self.z_encoder(x2)

        if self.n_genes > 0:
            x_genes = self.get_genes_batch(x_)
            ql_m, ql_v, library = self.l_encoder(x_genes)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()

            if self.n_genes > 0:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()

        if self.n_genes > 0:
            px_scale, px_r, px_rate, px_dropout = self.genes_decoder(
                "gene", z, library, batch_index, y
            )
            px_r = self.px_r
            px_r = torch.exp(px_r)

        if self.n_introns > 0:
            p = self.introns_decoder(z, self.first_indices)

        return (
            px_scale,
            px_r,
            px_rate,
            px_dropout,
            qz_m,
            qz_v,
            z,
            ql_m,
            ql_v,
            library,
            p,
        )

    def get_genes_batch(self, x):
        return x.narrow(1, 0, self.n_genes)

    def get_introns_batch(self, x):
        return x.narrow(1, self.n_genes, self.n_introns)

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        r"""Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution

        (
            px_scale,
            px_r,
            px_rate,
            px_dropout,
            qz_m,
            qz_v,
            z,
            ql_m,
            ql_v,
            library,
            p,
        ) = self.inference(x, batch_index, y)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        if self.n_introns > 0 and self.n_genes > 0:
            reconst_loss = self.weight_expression * (
                self.reconstruction_loss_genes(
                    self.get_genes_batch(x), px_rate, px_r, px_dropout
                )
                + kl(
                    Normal(ql_m, torch.sqrt(ql_v)),
                    Normal(local_l_mean, torch.sqrt(local_l_var)),
                ).sum(dim=1)
            ) + (
                self.weight_splicing
                * self.reconstruction_loss_introns(self.get_introns_batch(x), p)
            )
        elif self.n_introns > 0 and self.n_genes == 0:
            reconst_loss = self.reconstruction_loss_introns(
                self.get_introns_batch(x), p
            )
        elif self.n_introns == 0 and self.n_genes > 0:
            reconst_loss = (
                self.reconstruction_loss_genes(
                    self.get_genes_batch(x), px_rate, px_r, px_dropout
                )
                + kl(
                    Normal(ql_m, torch.sqrt(ql_v)),
                    Normal(local_l_mean, torch.sqrt(local_l_var)),
                ).sum(dim=1)
            )

        if self.regularization_gaussian_std is not None:
            weights = self.introns_decoder.first_part.weight
            kl_divergence -= (
                Normal(
                    torch.zeros_like(weights),
                    self.regularization_gaussian_std * torch.ones_like(weights),
                )
                .log_prob(weights)
                .sum()
            )

        if self.regularization_laplace_std is not None:
            weights = self.introns_decoder.first_part.weight
            kl_divergence -= (
                Laplace(
                    torch.zeros_like(weights),
                    self.regularization_laplace_std * torch.ones_like(weights),
                )
                .log_prob(weights)
                .sum()
            )

        return reconst_loss, kl_divergence, 0.0


class UnsupervisedTrainer(UnsupervisedTrainer_scVI):
    def __init__(
        self,
        model,
        gene_dataset,
        train_size=0.8,
        test_size=None,
        n_epochs_kl_warmup=400,
        **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        if type(self) is UnsupervisedTrainer:
            (
                self.train_set,
                self.test_set,
                self.validation_set,
            ) = self.train_test_validation(
                model, gene_dataset, train_size, test_size, type_class=Posterior
            )
            self.train_set.to_monitor = ["elbo"]
            self.test_set.to_monitor = ["elbo"]
            self.validation_set.to_monitor = ["elbo"]


def compute_reconstruction_error_2(vae, posterior, **kwargs):
    elbo = 0
    for i_batch, tensors in enumerate(posterior):
        sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors[
            :5
        ]  # general fish case
        reconst_loss, kl_divergence, _ = vae(
            sample_batch,
            local_l_mean,
            local_l_var,
            batch_index=batch_index,
            y=labels,
            **kwargs
        )
        elbo += torch.sum(reconst_loss).item()
    n_samples = len(posterior.indices)
    return elbo / n_samples


class Posterior(scVIPosterior):
    @torch.no_grad()
    def reconstruction_error(self):
        reconstruction_error = compute_reconstruction_error_2(self.model, self)
        print("reconstruction_error: ", reconstruction_error)
        return reconstruction_error

    @torch.no_grad()
    def get_z_m_v(self):
        M = []
        V = []
        for tensors in self:
            sample_batch = tensors[0]
            # m, v, _ = self.model.z_encoder(sample_batch)
            m, v, _ = self.model.z_encoder(self.model.transform_input(sample_batch))
            M.append(m)
            V.append(v)
        return (torch.cat(M), torch.cat(V))

    @torch.no_grad()
    def calculate_lfc_psi_and_bf(
        self,
        cell_idx_a,
        cell_idx_b=None,
        N_z_samples=5,
        batch_size=128,
        N_batches=10,
        mode="change",
        lfc_psi_threshold=0.5,
    ):
        dataset = self.gene_dataset
        if cell_idx_b is None:
            cell_idx_b = np.delete(np.arange(dataset.n_cells), cell_idx_a)

        zM, zV = self.get_z_m_v()

        zM_a = zM[cell_idx_a]
        zV_a = zV[cell_idx_a]

        zM_b = zM[cell_idx_b]
        zV_b = zV[cell_idx_b]

        lfc_psi_accum = np.zeros(dataset.n_introns, dtype=float)
        lfc_psi_significant_accum = np.zeros(dataset.n_introns, dtype=float)

        def get_p(batch_idx, zM, zV):
            zM_batch = (
                zM[batch_idx]
                .expand(N_z_samples, -1, -1)
                .reshape(-1, self.model.n_latent)
            )
            zV_batch = (
                zV[batch_idx]
                .expand(N_z_samples, -1, -1)
                .reshape(-1, self.model.n_latent)
            )
            z = Normal(zM_batch, zV_batch.sqrt()).sample()
            p = self.model.introns_decoder(z, self.model.first_indices)
            return p

        for i in range(N_batches):
            if i % 1000 == 0:
                print(i)
            batch_idx_a = np.random.randint(len(cell_idx_a), size=batch_size)
            p_a = get_p(batch_idx_a, zM_a, zV_a)

            batch_idx_b = np.random.randint(len(cell_idx_b), size=batch_size)
            p_b = get_p(batch_idx_b, zM_b, zV_b)

            epsilon = 1e-4
            lfc_psi = torch.log2(p_a + epsilon) - torch.log2(p_b + epsilon)
            if mode == "change":
                lfc_psi_significant = (lfc_psi < -lfc_psi_threshold) | (
                    lfc_psi > lfc_psi_threshold
                )
            else:
                lfc_psi_significant = lfc_psi > 0
            lfc_psi = lfc_psi.float().mean(dim=0).cpu().numpy()
            lfc_psi_significant = lfc_psi_significant.float().mean(dim=0).cpu().numpy()
            lfc_psi_accum += lfc_psi
            lfc_psi_significant_accum += lfc_psi_significant

        lfc_psi = lfc_psi_accum / N_batches
        lfc_psi_significant = lfc_psi_significant_accum / N_batches
        bf = np.log(lfc_psi_significant + 1e-8) - np.log(1 - lfc_psi_significant + 1e-8)
        return lfc_psi, bf


def run_vae(
    adata,
    n_epochs=300,
    use_cuda=True,
    n_latent=20,
    n_layers=1,
    dropout_rate = 0.25,
    n_hidden = 128,
    lr = 1e-2,
    n_epochs_kl_warmup = 20,
    linearity = 'linear',
    loss_introns = "dirichlet-multinomial",
    input_transform = "log",
    regularization_gaussian_std = None,
    sample=True,
    feature_addition=None,
):
    dataset = Dataset(adata)
    print(dataset.X.shape)

    early_stopping = dict(
        #early_stopping_metric='elbo',
        early_stopping_metric='reconstruction_error',
        on="test_set",
        patience=10,  #20
        threshold=1,
        reduce_lr_on_plateau=True,
        lr_patience=5,  #10
        lr_factor= 0.5,
       )

    vae = VAE(dataset.n_genes, dataset.n_introns, dataset.n_clusters,
              dataset.intron_clusters, n_latent=n_latent, n_layers=n_layers,
              n_hidden=n_hidden, dropout_rate=dropout_rate, use_cuda=use_cuda,
              loss_introns=loss_introns, linearly_decoded=linearity=='linear',
              input_transform=input_transform, feature_addition=feature_addition,
              linearly_encoded=False, regularization_gaussian_std=regularization_gaussian_std)

    trainer = UnsupervisedTrainer(
        vae, dataset, train_size=0.9, test_size=0.1, use_cuda=use_cuda, frequency=5,
        n_epochs_kl_warmup=n_epochs_kl_warmup, early_stopping_kwargs=early_stopping)

    trainer.train(n_epochs=n_epochs, lr=lr)
    vae.eval()
    posterior = Posterior(vae, dataset, use_cuda=use_cuda, data_loader_kwargs={'batch_size': 128})
    latent = posterior.get_latent(sample=sample)[0]
    return latent, trainer.model
