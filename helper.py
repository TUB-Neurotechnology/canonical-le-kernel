import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea
import sklearn
from scipy.linalg import eigh, solve
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

from moabb.analysis.meta_analysis import collapse_session_scores
import pyriemann as pr
from pyriemann.utils.base import logm, invsqrtm
from pyriemann.utils.distance import pairwise_distance
from pyriemann.utils.kernel import kernel as kernel_fct
from pyriemann.utils.mean import mean_logeuclid, mean_riemann, mean_covariance, mean_functions



def kernel_cle(X, Y=None, *, Cref=None, metric='nada', reg=1e-10):
    r""" Canonical Log-Euclidean kernel between two sets of SPD matrices.

    Calculates the Canonical Log-Euclidean kernel matrix :math:`\mathbf{K}` of
    inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products given a reference matrix :math:`\mathbf{C}_{\text{ref}}` [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}((\log(\mathbf{X}_i) -
        \log(\mathbf{C}_{\text{ref}}))
        (\log(\mathbf{Y}_j) - \log(\mathbf{C}_{\text{ref}})))

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference matrix. If None, the mean of Y is used.
    metric : str, default='nada'
        Just to match the signature of the kernel function.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Canonical Log-Euclidean kernel matrix between X and Y.
    """
    if Cref is None:
        if Y is None:
            Cref = pr.utils.mean.mean_logeuclid(X)
        else:
            Cref = pr.utils.mean.mean_logeuclid(Y)

    def kernelfct(X, Cref):
        logCref = logm(Cref)
        X_ = logm(X)
        X_ = X_ - logCref
        return X_

    return pr.utils.kernel._apply_matrix_kernel(kernelfct, X, Y, Cref=Cref,
                                                reg=reg)


def kernel_riemann_online(X, Y=None, *, Cref=None, metric='riemann',
                          n_samples_mean=0.2, reg=1e-10):
    r"""Online kernel between two sets of SPD matrices according to AIR metric.

    Calculates the online kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products given
    a reference matrix :math:`\mathbf{C}_{\text{ref}}`. The reference matrix
    is computed as the mean of `n_samples_mean` matrices and updated
    at each iteration [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Just to match the signature of the kernel function.
    metric : str, default='riemann'
        Just to match the signature of the kernel function.
    n_samples_mean : float or int, default=0.2
        Number of samples to compute the reference matrix. If float, it
        represents the proportion of the number of matrices in X.
        If int, it represents the number of matrices.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The online kernel matrix between X and Y.

    """
    n_matrices_X, n_channels, n_channels = X.shape
    arrays_equal = np.array_equal(X, Y)
    if arrays_equal:
        if n_samples_mean < 1:
            n_samples_mean = int(n_samples_mean * n_matrices_X)
        Crefs_start = np.repeat(
            invsqrtm(mean_riemann(X[:n_samples_mean]))[None, ...],
            n_samples_mean, axis=0)

        Crefs = invsqrtm(np.array(
            [mean_riemann(X[i:i + n_samples_mean]) for i in
             range(len(X) - n_samples_mean)]))
        CrefsX = np.vstack((Crefs_start, Crefs))
        X_ = CrefsX @ X @ CrefsX
        X_ = logm(X_)
        Y_ = X_

    else:
        n_matrices_Y, n_channels, n_channels = Y.shape
        if n_samples_mean < 1:
            n_samples_mean = int(n_samples_mean * n_matrices_Y)
        Crefs_start = np.repeat(
            invsqrtm(mean_riemann(Y[:n_samples_mean]))[None, ...],
            n_samples_mean, axis=0)

        YX = np.vstack((Y, X))
        Crefs = invsqrtm(np.array(
            [mean_riemann(YX[i:i + n_samples_mean]) for i in
             range(len(YX) - n_samples_mean)]))

        CrefsYX = np.vstack((Crefs_start, Crefs))
        CrefsY = CrefsYX[:n_matrices_Y]
        CrefsX = CrefsYX[n_matrices_Y:]

        X_ = CrefsX @ X @ CrefsX
        X_ = logm(X_)
        Y_ = CrefsY @ Y @ CrefsY
        Y_ = logm(Y_)

    K = np.einsum('abc,dbc->ad', X_, Y_, optimize=True)

    # regularization due to numerical errors
    if arrays_equal:
        K.flat[:: n_matrices_X + 1] += reg

    return K


def kernel_cle_online(X, Y=None, *, Cref=None, metric='cle', n_samples_mean=0.2,
                      reg=1e-10):
    r"""Online Canonical Log-Euclidean kernel between two sets of SPD matrices.

    Calculates the online Canonical Log-Euclidean kernel matrix :math:`\mathbf{K}`
    of inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise products
    given a reference matrix :math:`\mathbf{C}_{\text{ref}}`. The reference matrix
    is computed as the mean of `n_samples_mean` matrices and updated at each iteration.


    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Just to match the signature of the kernel function.
    metric : str, default='cle'
        Just to match the signature of the kernel function.
    n_samples_mean : float or int, default=0.2
        Number of samples to compute the reference matrix. If float, it
        represents the proportion of the number of matrices in X.
        If int, it represents the number of matrices.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The online Canonical Log-Euclidean kernel matrix between X and Y.

    """
    n_matrices_X, n_channels, n_channels = X.shape
    arrays_equal = np.array_equal(X, Y)
    if arrays_equal:
        if n_samples_mean < 1:
            n_samples_mean = int(n_samples_mean * n_matrices_X)
        Crefs_start = np.repeat(
            logm(mean_logeuclid(X[:n_samples_mean]))[None, ...], n_samples_mean,
            axis=0)

        Crefs = logm(np.array([mean_logeuclid(X[i:i + n_samples_mean]) for i in
                               range(len(X) - n_samples_mean)]))
        CrefsX = np.vstack((Crefs_start, Crefs))

        X_ = logm(X) - CrefsX

        Y_ = X_


    else:
        n_matrices_Y, n_channels, n_channels = Y.shape
        if n_samples_mean < 1:
            n_samples_mean = int(n_samples_mean * n_matrices_Y)
        Crefs_start = np.repeat(
            logm(mean_logeuclid(Y[:n_samples_mean]))[None, ...], n_samples_mean,
            axis=0)

        YX = np.vstack((Y, X))
        Crefs = logm(np.array([mean_logeuclid(YX[i:i + n_samples_mean]) for i in
                               range(len(YX) - n_samples_mean)]))

        all_Cref = np.vstack((Crefs_start, Crefs))
        CrefsY = all_Cref[:n_matrices_Y]
        CrefsX = all_Cref[n_matrices_Y:]

        X_ = logm(X) - CrefsX
        Y_ = logm(Y) - CrefsY

    K = np.einsum('abc,dbc->ad', X_, Y_, optimize=True)

    # regularization due to numerical errors
    if arrays_equal:
        K.flat[:: n_matrices_X + 1] += reg
    return K


def kernel_gle(X, Y=None, *, gamma=0.001, Cref=None, metric='cle'):
    r"""Gaussian Log-Euclidean kernel between two sets of SPD matrices.

    Calculates the Gaussian Log-Euclidean kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices
    in :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-\gamma ||\log(\mathbf{X}_i) - \log(\mathbf{Y}_j)||^2)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    gamma : float, default=0.001
        Regularization parameter.
    Cref : None | ndarray, shape (n, n), default=None
        Just to match the signature of the kernel function.
    metric : str, default='cle'
        Just to match the signature of the kernel function.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Gaussian Log-Euclidean kernel matrix between X and Y.
    """
    if np.array_equal(X, Y):
        Y = None

    pairwise_dist_squared = pairwise_distance(X, Y, metric='logeuclid',
                                              squared=True)
    K = np.exp(-pairwise_dist_squared * gamma)

    return K


pr.utils.kernel.kernel_cle = kernel_cle
pr.utils.mean.mean_functions = {**pr.utils.mean.mean_functions,
                                'cle': pr.utils.mean.mean_logeuclid}

pr.utils.distance.distance_functions = {**pr.utils.distance.distance_functions,
                                        'cle': pr.utils.distance.distance_logeuclid}


class Gram(BaseEstimator, TransformerMixin):
    r"""Gram matrix transformer for kernel functions.

    This transformer computes the Gram matrix between two sets of SPD matrices
    using a kernel function. The kernel function is used to compute the inner
    product between the matrices in the two sets. The kernel function is
    specified by the user and can be any of the available kernel functions in
    :mod:`pyriemann.utils.kernel` or a custom function.
    The Gram matrix of a kernel function ``k`` between two sets of SPD matrices
    X and Y is defined as:

    .. math::
        \mathbf{K}_{i,j} = \text{k}(\mathbf{X}_i, \mathbf{Y}_j)

    Parameters
    ----------
    metric : str
        The metric to use to compute the mean. See
        :func:`pyriemann.utils.mean.mean_covariance` for available options.
    kernel_fct : callable
        The kernel to use to compute the gram matrix. See
        :func:`pyriemann.utils.kernel.kernel` for available options.
    kernel_params : dict
        Parameters to pass to the kernel function.

    Attributes
    ----------
    data_ : ndarray, shape (n_trials, n_channels, n_channels)
        The data used to compute the mean covariance matrix.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference covariance matrix.

    See Also
    --------
    pyriemann.utils.kernel.kernel

    """

    def __init__(self, metric, kernel_fct, kernel_params=None, Cref=None):
        self.metric = metric
        self.kernel_fct = kernel_fct
        self.kernel_params = kernel_params
        self.Cref = Cref

    def fit(self, X, y=None):
        self.data_ = X
        if self.Cref is None and self.metric in mean_functions.keys():
            self.Cref = mean_covariance(X, metric=self.metric)
        if self.kernel_params is None:
            self.kernel_params = {}
        return self

    def transform(self, X, y=None):
        gram = self.kernel_fct(X, self.data_,
                               Cref=self.Cref,
                               **self.kernel_params)
        return gram


class GramRuntimes(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Gram matrix transformer.

    Parameters
    ----------
    metric : str
        The metric to use to compute the mean. See
        :func:`pyriemann.utils.mean.mean_covariance` for available options.
    kernel : str
        The kernel to use to compute the gram matrix. See
        :func:`pyriemann.utils.kernel.kernel` for available options.

    Attributes
    ----------
    data_ : ndarray, shape (n_trials, n_channels, n_channels)
        The data used to compute the mean covariance matrix.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference covariance matrix.

    See Also
    --------
    pyriemann.utils.mean.mean_covariance
    pyriemann.utils.kernel.kernel

    """

    def __init__(self, metric, kernel):
        self.metric = metric
        self.kernel = kernel

    def fit(self, X, y=None):
        self.data_ = X
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'data_'):
            self.data_ = X
        gram = self.kernel(X, self.data_)
        return gram

    def fit_transform(self, X, y=None):
        gram = self.fit(X, y).transform(X, y)

        return gram


def _simplify_names(x):
    if len(x) > 10:
        return x.split(" ")[0]
    else:
        return x


def QNX(N, k, sorted_dist_hd, sorted_dist_ld):
    r"""Quality of Nearest Neighbors (QNX) metric.

    The QNX metric is a measure of the quality of the nearest neighbors in the
    low-dimensional space. It is defined as the ratio of the number of common
    nearest neighbors in the high-dimensional space and the low-dimensional
    space divided by the number of neighbors defined as:

    .. math::
        \text{QNX} = \frac{1}{Nk} \sum_{i=1}^{N} |kNN(X_{hd}) \cap kNN(X_{ld})|


    Parameters
    ----------
    N : int
        The number of samples.
    k : int
        The number of neighbors to consider.
    sorted_dist_hd : ndarray, shape (n_samples, n_samples)
        The sorted distance matrix of the high-dimensional data.
    sorted_dist_ld : ndarray, shape (n_samples, n_samples)
        The sorted distance matrix of the low-dimensional data.

    Returns
    -------
    res : float
        The QNX metric.

    """
    res = np.sum([len(np.intersect1d(i, j)) for i, j in
                  zip(sorted_dist_hd, sorted_dist_ld)]) / (k * N)
    return res


def RNX(N, k, sorted_dist_hd, sorted_dist_ld):
    r"""Ratio of Nearest Neighbors (RNX) metric.

    The RNX metric is a measure of the ratio of the number of common nearest
    neighbors in the high-dimensional space and the low-dimensional space
    divided by the number of neighbors defined as:

    .. math::
        \text{RNX} = \frac{(N-1)QNX - k}{N-1-k}

    Parameters
    ----------
    N : int
        The number of samples.
    k : int
        The number of neighbors to consider.
    sorted_dist_hd : ndarray, shape (n_samples, n_samples)
        The sorted distance matrix of the high-dimensional data.
    sorted_dist_ld : ndarray, shape (n_samples, n_samples)
        The sorted distance matrix of the low-dimensional data.

    Returns
    -------
    res : float
        The RNX metric.

    """
    qnx = QNX(N, k, sorted_dist_hd, sorted_dist_ld)
    return ((N - 1) * qnx - k) / (N - 1 - k)


def AUClnK(data, embd, metric, n_jobs=8):
    r"""Area Under the Curve of the Ratio of Nearest Neighbors (AUC-RNN) metric.

    The AUC-RNN metric is a measure of the quality of the embedding. It is
    defined as the area under the curve of the ratio of the number of common
    nearest neighbors in the high-dimensional space and the low-dimensional
    space divided by the number of neighbors defined as:

    .. math::
        \text{AUC-RNN} = \frac{1}{N-2} \sum_{k=1}^{N-2} \frac{RNX(k)}{k}

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        The high-dimensional data.
    embd : ndarray, shape (n_samples, n_components)
        The low-dimensional data.
    metric : str
        The metric to use to compute the distance matrix. See
        :func:`pyriemann.utils.distance.pairwise_distance` for available options.
    n_jobs : int, default=8
        The number of jobs to run in parallel.

    Returns
    -------
    res : float
        The AUC-RNN metric.

    """

    N = len(data)
    if metric == 'cle':
        metric = 'logeuclid'
    pairwise_distance_hd = pr.utils.distance.pairwise_distance(data,
                                                               metric=metric)
    sorted_dist_hd = np.argsort(pairwise_distance_hd)
    # get knn embd
    pairwise_distance_ld = distance_matrix(embd, embd)
    sorted_dist_ld = np.argsort(pairwise_distance_ld)

    if n_jobs == 1:
        print('not parallel')
        nom = np.sum([RNX(N, k, sorted_dist_hd[:, :k],
                          sorted_dist_ld[:, :k]) / k for k in range(1, N - 2)])
    else:
        def parrnk(k):
            return RNX(N, k, sorted_dist_hd[:, :k],
                       sorted_dist_ld[:, :k]) / k

        nom = np.sum(Parallel(n_jobs=n_jobs)(
            delayed(parrnk)(k) for k in range(1, N - 2)))
    denom = np.sum([1 / k for k in range(1, N - 2)])
    return nom / denom


class AUClnKWrapper(sklearn.base.BaseEstimator):
    """
    AUClnK wrapper for sklearn.
    
    Parameters
    ----------
    embedding : object
        The embedding object to use.
    metric : str
        The metric to use to compute the distance matrix. See
        :func:`pyriemann.utils.distance.pairwise_distance` for available options.
    n_jobs : int, default=8
        The number of jobs to run in parallel.
    """
    def __init__(self, embedding, metric, n_jobs=8):
        self.embedding = embedding
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        embd = self.embedding.fit_transform(X)
        score = AUClnK(X, embd, self.metric, self.n_jobs)
        return score

    def predict(X, y):
        return np.ones(len(y))

