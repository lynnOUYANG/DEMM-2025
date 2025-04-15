# # Author: Kyohei Atarashi
# # License: BSD-2-Clause
#
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils import check_random_state, check_array
# from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.extmath import safe_sparse_dot
# from math import sqrt
# # from .utils import get_random_matrix, next_pow_of_two
# from scipy.linalg import qr_multiply
# from scipy.stats import chi
# import warnings
# import math
# def standard_gaussian(random_state, size):
#     return random_state.randn(*size)
#
#
# def rademacher(random_state, size):
#     return random_state.randint(2, size=size, dtype=np.int32)*2-1
#
#
# def laplace(random_state, size):
#     return random_state.laplace(0, 1. / np.sqrt(2), size)
#
#
# def uniform(random_state, size):
#     return random_state.uniform(-np.sqrt(3), np.sqrt(3), size)
# def get_random_matrix(random_state, distribution, size, p_sparse=0.,
#                       dtype=np.float64):
#     # size = (n_components, n_features)
#     if distribution == 'rademacher':
#         return rademacher(random_state, size).astype(dtype)
#     elif distribution in ['gaussian', 'normal']:
#         return standard_gaussian(random_state, size)
#     elif distribution == 'uniform':
#         return uniform(random_state, size)
#     elif distribution == 'laplace':
#         return laplace(random_state, size)
#     # elif distribution == 'sparse_rademacher':
#     #     # n_nzs : (n_features, )
#     #     # n_nzs[j] is n_nz of random_weights[:, j]
#     #     return sparse_rademacher(random_state, np.array(size, dtype=np.int32),
#     #                              p_sparse)
#     else:
#         raise ValueError('{} distribution is not implemented. Please use'
#                          'rademacher, gaussian (normal), uniform or laplace.'
#                          .format(distribution))
# def _get_random_matrix(distribution):
#     return lambda rng, size: get_random_matrix(rng, distribution, size)
#
#
# class OrthogonalRandomFeature(BaseEstimator, TransformerMixin):
#     """Approximates feature map of the RBF or dot kernel
#     by Orthogonal Random Feature map.
#
#     Parameters
#     ----------
#     n_components : int (default=100)
#         Number of Monte Carlo samples per original features.
#         Equals the dimensionality of the computed (mapped) feature space.
#         If n_components is not a n-tuple of n_features, it is automatically
#         changed to the smallest n-tuple of the n_features that is bigger than
#         n_features, which is bigger than n_components.
#         That is, ceil(n_components/n_features)*n_features.
#
#     gamma : float (default=0.5)
#         Bandwidth parameter. gamma = 1/2\sigma^2, where \sigma is a std
#         parameter for the Gaussian distribution.
#
#     distribution : str or function (default="gaussian")
#         A function for sampling random bases.
#         Its arguments must be random_state and size.
#         For str, "gaussian" (or "normal"), "rademacher", "laplace", or
#         "uniform" can be used.
#
#     random_fourier : boolean (default=True)
#         Whether to approximate the RBF kernel or not.
#         If True, this class samples random_offset_ in the fit method and
#         computes the cosine of structured_matrix-feature_vector product
#         + random_offset_ in transform.
#         If False, OrthogonalRandomFeature does not sample it and computes just
#         structured_matrix-feature_vector product (i.e., approximates dot
#         product kernel).
#
#     use_offset : bool (default=False)
#         If True, Z(x) = (cos(w_1x+b_1), cos(w_2x+b_2), ... , cos(w_Dx+b_D),
#         where w is random_weights and b is offset (D=n_components).
#         If False, Z(x) = (cos(w_1x), ..., cos(w_{D/2}x), sin(w_1x), ...,
#         sin(w_{D/2}x)).
#
#     random_state : int, RandomState instance or None, optional (default=None)
#         If int, random_state is the seed used by the random number generator;
#         If np.RandomState instance, random_state is the random number generator;
#         If None, the random number generator is the RandomState instance used
#         by `np.random`.
#
#     Attributes
#     ----------
#     random_weights_ : array, shape (n_features, n_components) (use_offset=True)
#     or (n_components/2, n_features) (otherwise)
#         The sampled basis.
#
#     random_offset_ : array or None, shape (n_components, )
#         The sampled offset vector. If use_offset=False, random_offset_=None.
#
#     References
#     ----------
#     [1] Orthogonal Random Features.
#     Felix Xinnan Yu, Ananda Theertha Suresh, Krzysztof Choromanski,
#     Daniel Holtmann-Rice, and Sanjiv Kumar.
#     In NIPS 2016.
#     (https://arxiv.org/pdf/1610.09072.pdf)
#
#     """
#
#     def __init__(self, n_components=100, gamma=0.5, distribution="gaussian",
#                  random_fourier=True, use_offset=False, random_state=None):
#         self.n_components = n_components
#         self.distribution = distribution
#         self.gamma = gamma
#         self.random_fourier = random_fourier
#         self.use_offset = use_offset
#         self.random_state = random_state
#
#     def fit(self, X, y=None):
#         """Generate random weights according to n_features.
#
#         Parameters
#         ----------
#         X : {array-like, sparse matrix}, shape (n_samples, n_features)
#             Training data, where n_samples is the number of samples
#             and n_features is the number of features.
#
#         Returns
#         -------
#         self : object
#             Returns the transformer.
#         """
#         random_state = check_random_state(self.random_state)
#         X = check_array(X, accept_sparse=True)
#         n_samples, n_features = X.shape
#         n_stacks = int(np.ceil(self.n_components / n_features))
#         n_components = n_stacks * n_features
#         if n_components != self.n_components:
#             msg = "n_components is changed from {0} to {1}.".format(
#                 self.n_components, n_components
#             )
#             msg += " You should set n_components to an n-tuple of n_features."
#             warnings.warn(msg)
#             self.n_components = n_components
#
#         if self.random_fourier and not self.use_offset:
#             n_stacks = int(np.ceil(n_stacks / 2))
#             n_components = n_stacks * n_features
#             if n_components * 2 != self.n_components:
#                 msg = "n_components is changed from {0} to {1}.".format(
#                     self.n_components, n_components * 2
#                 )
#                 msg += " When random_fourier=True and use_offset=False, "
#                 msg += " n_components should be larger than 2*n_features."
#                 warnings.warn(msg)
#                 self.n_components = n_components * 2
#
#         if self.gamma == 'auto':
#             gamma = 1.0 / X.shape[1]
#         else:
#             gamma = self.gamma
#
#         size = (n_features, n_features)
#         if isinstance(self.distribution, str):
#             distribution = _get_random_matrix(self.distribution)
#         else:
#             distribution = self.distribution
#         random_weights_ = []
#         for _ in range(n_stacks):
#             W = distribution(random_state, size)
#             S = np.diag(chi.rvs(df=n_features, size=n_features,
#                                 random_state=random_state))
#             SQ, _ = qr_multiply(W, S)
#             random_weights_ += [SQ]
#
#         self.random_weights_ = np.vstack(random_weights_).T
#         self.random_offset_ = None
#         if self.random_fourier:
#             self.random_weights_ *= sqrt(2 * gamma)
#             if self.use_offset:
#                 self.random_offset_ = random_state.uniform(
#                     0, 2 * np.pi, size=n_components
#                 )
#
#         return self
#
#     def transform(self, X):
#         """Apply the approximate feature map to X.
#
#         Parameters
#         ----------
#         X : {array-like, sparse matrix}, shape (n_samples, n_features)
#             New data, where n_samples is the number of samples
#             and n_features is the number of features.
#
#         Returns
#         -------
#         X_new : array-like, shape (n_samples, n_components)
#         """
#         check_is_fitted(self, "random_weights_")
#         X = check_array(X, accept_sparse=True)
#
#         output = safe_sparse_dot(X, self.random_weights_, dense_output=True)
#         if self.random_fourier:
#             if self.use_offset:
#                 output = np.cos(output + self.random_offset_)
#             else:
#                 output = np.hstack((np.cos(output), np.sin(output)))
#             output *= np.sqrt(2)
#         print(self.n_components)
#         return output / sqrt(self.n_components)
#
#     def _remove_bases(self, indices):
#         if self.random_fourier and not self.use_offset:
#             warnings.warn("Bases are not removed when use_offset=False and"
#                           " random_fourier=True.")
#             return False
#         else:
#             self.random_weights_ = np.delete(self.random_weights_, indices, 1)
#             if self.random_fourier:
#                 self.random_offset_ = np.delete(self.random_offset_, indices, 0)
#             self.n_components = self.random_weights_.shape[1]
#             return True
import time

import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from math import sqrt
from scipy.stats import chi
import warnings


def standard_gaussian(random_state, size,seed):
    torch_rng = torch.Generator()
    torch_rng.manual_seed(random_state.randint(seed))
    return torch.randn(size, dtype=torch.float32, generator=torch_rng)


def rademacher(random_state, size):
    torch_rng = torch.Generator()
    torch_rng.manual_seed(random_state.randint(232 - 1))
    return (torch.randint(0, 2, size=size, dtype=torch.int32, generator=torch_rng) * 2 - 1)


def laplace(random_state, size):
    return torch.empty(size).exponential_(1. / torch.sqrt(torch.tensor(2.))) - 1


def uniform(random_state, size):
    return (torch.empty(size).uniform_(-torch.sqrt(torch.tensor(3.)), torch.sqrt(torch.tensor(3.))))


def get_random_matrix(random_state, distribution, size, p_sparse=0., dtype=torch.float64,seed=0):
    if distribution == 'rademacher':
        return rademacher(random_state, size).type(dtype)
    elif distribution in ['gaussian', 'normal']:
        return standard_gaussian(random_state, size,seed)
    elif distribution == 'uniform':
        return uniform(random_state, size)
    elif distribution == 'laplace':
        return laplace(random_state, size)
    else:
        raise ValueError('{} distribution is not implemented. Please use'
                         ' rademacher, gaussian (normal), uniform or laplace.'
                         .format(distribution))


def _get_random_matrix(distribution,seed):
    return lambda rng, size: get_random_matrix(rng, distribution, size,seed=seed)


class OrthogonalRandomFeature(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, gamma=0.5, distribution="gaussian",
                 random_fourier=True, use_offset=False, random_state=None):
        self.n_components = n_components
        self.distribution = distribution
        self.gamma = gamma
        self.random_fourier = random_fourier
        self.use_offset = use_offset
        self.random_state = random_state

    def fit(self, X, y=None,seed=0):
        device=X.device

        random_state = check_random_state(self.random_state)

        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor.")

        n_samples, n_features = X.shape
        n_stacks = int(torch.ceil(torch.tensor(self.n_components / n_features)))
        n_components = n_stacks * n_features
        if n_components != self.n_components:
            msg = "n_components is changed from {0} to {1}.".format(self.n_components, n_components)
            msg += " You should set n_components to an n-tuple of n_features."
            warnings.warn(msg)
            self.n_components = n_components

        if self.random_fourier and not self.use_offset:
            n_stacks = int(torch.ceil(torch.tensor(n_stacks / 2)))
            n_components = n_stacks * n_features
            if n_components * 2 != self.n_components:
                msg = "n_components is changed from {0} to {1}.".format(self.n_components, n_components * 2)
                msg += " When random_fourier=True and use_offset=False, n_components should be larger than 2*n_features."
                warnings.warn(msg)
                self.n_components = n_components * 2

        if self.gamma == 'auto':
            gamma = 1.0 / n_features
        else:
            gamma = self.gamma

        size = (n_features, n_features)
        if isinstance(self.distribution, str):
            distribution = _get_random_matrix(self.distribution,seed)
        else:
            distribution = self.distribution
        random_weights_ = []
        for _ in range(n_stacks):
            W = distribution(random_state, size).to(torch.float64)
            S = torch.diag(
                torch.tensor(chi.rvs(df=n_features, size=n_features, random_state=random_state), dtype=torch.float64))
            Q,R=torch.linalg.qr(W)
            SQ=torch.matmul(Q,S)
            random_weights_.append(SQ)

        self.random_weights_ = torch.vstack(random_weights_).T
        # self.random_weights_=self.random_weights_.to(device).to(torch.float64)
        self.random_weights_ = self.random_weights_.to(device).to(torch.float32)
        self.random_offset_ = None
        if self.random_fourier:
            self.random_weights_ *= sqrt(2 * gamma)
            if self.use_offset:
                self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=n_components)
                self.random_offset_ = torch.tensor(self.random_offset_, dtype=torch.float64)

        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor.")
        X = X.to(torch.float32)
        # print(X.device)
        # matrix = torch.randn((3025, 128), dtype=torch.float32, device="cuda:2")

        st_time=time.time()

        output = torch.matmul(X, self.random_weights_)
        # D=torch.matmul(matrix,matrix.T)
        en_time=time.time()-st_time
        print(en_time)
        # exit()
        if self.random_fourier:
            if self.use_offset:
                output = torch.cos(output + self.random_offset_)
            else:
                output = torch.cat((torch.cos(output), torch.sin(output)), dim=1)
            output *= torch.sqrt(torch.tensor(2.0))

        return output / sqrt(self.n_components)

    def _remove_bases(self, indices):
        if self.random_fourier and not self.use_offset:
            warnings.warn("Bases are not removed when use_offset=False and random_fourier=True.")
            return False
        else:
            mask = torch.ones(self.random_weights_.size(1), dtype=torch.bool)
            mask[indices] = False
            self.random_weights_ = self.random_weights_[:, mask]
            if self.random_fourier:
                self.random_offset_ = self.random_offset_[mask]
            self.n_components = self.random_weights_.size(1)
            return True
