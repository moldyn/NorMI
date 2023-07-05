# -*- coding: utf-8 -*-
"""Class for estimating normalized mutual information.

MIT License
Copyright (c) 2023, Daniel Nagel
All rights reserved.

"""
__all__ = ['NormalizedMI']  # noqa: WPS410

import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple, Optional, Union
from scipy.special import digamma
from scipy.spatial import KDTree, kdtree
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from nmi._typing import (  # noqa: WPS436
    ArrayLikeFloat,
    Float2DArray,
    FloatMax2DArray,
    NormalizedMatrix,
    NormString,
    PositiveMatrix,
    PositiveInt,
)


class NormalizedMI(BaseEstimator):
    r"""Class for estimating the normalized mutual information.

    Parameters
    ----------
    n_dim : int, default=1
        Dimensionality of input vectors.
    normalize_method : str, default='joint'
        Determines the normalization factor for the mutual information:
        - `'joint'` is the joint entropy
        - `'max'` is the maximum of the individual entropies
        - `'arithmetic'` is the mean of the individual entropies
        - `'geometric'` is the square root of the product of the individual
          entropies
        - `'min'` is the minimum of the individual entropies

    Attributes
    ----------
    mi_ : ndarray of shape (n_features, n_features)
        The pairwise mutual information matrix of the data.
    nmi_ : ndarray of shape (n_features, n_features)
        The normalized pairwise mutual information matrix of the data.
    hxy_ : ndarray of shape (n_features, n_features)
        The pairwise joint entropy matrix of the data.
    hx_ : ndarray of shape (n_features, n_features)
        The pairwise entropy matrix of the data.
    hy_ : ndarray of shape (n_features, n_features)
        The pairwise entropy matrix of the data.

    Examples
    --------
    >>> from nmi import NormalizedMI
    >>> x = np.linspace(0, np.pi, 1000)
    >>> data = np.array([np.cos(x), np.cos(x + np.pi / 6)]).T
    >>> nmi = NormalizedMI()
    >>> nmi.fit(data)
    NormalizedMI()
    >>> nmi.nmi_
    array([[1.       , 0.9697832],
           [0.9697832, 1.       ]])

    """

    _dtype: np.dtype = np.float64
    _default_normalize_method: str = 'joint'
    _default_n_dim: int = 1
    # todo add to __init__
    k = 5
    invariant_measure = 'radii'

    @beartype
    def __init__(
        self,
        *,
        n_dim: Optional[PositiveInt] = None
        normalize_method: Optional[NormString] = None,
    ):
        """Initialize NormalizedMI class."""
        if normalize_method is None:
            normalize_method = self._default_normalize_method
        if n_dim is None:
            n_dim = self._default_n_dim
        self.normalize_method: NormString = normalize_method
        self.n_dim: PositiveInt = n_dim

    @beartype
    def fit(
        self,
        X: FloatMax2DArray,
        y: Optional[ArrayLikeFloat] = None,
    ):
        """Compute the normalized mutual information matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features x n_dim)
            Training data.
        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        self._reset()


        # parse data
        if X.ndim < 2 * self.n_dim:
            raise ValueError('At least two variables need to be provided')
        stds = np.std(X, axis=0)
        if np.any(stds == 0) or np.any(np.isnan(stds)):
            idxs = np.where((stds == 0) | (np.isnan(stds)))[0]
            raise ValueError(
                f'Columns {idxs} have a standard deviation of zero or NaN. '
                'These columns cannot be used for estimating the NMI.'
            )

        # define number of features and samples
        n_samples: int
        n_cols: int
        n_features: int
        n_samples, n_cols = X.shape
        n_features = n_cols // self.n_dim

        if n_cols != n_features * self.n_dim:
            raise ValueError(
                'The number of provided columns needs to be a multiple of the '
                'specified dimensionality `n_dim`.'
            )

        self._n_samples: int = n_samples
        self._n_features: int = n_features

        # scale input
        X = StandardScaler(copy=False).fit_transform(X)

        self.mi_: PositiveMatrix
        self.hxy_: PositiveMatrix
        self.hx_: PositiveMatrix
        self.hy_: PositiveMatrix

        self.mi_, self.hxy_, self.hx_, self.hy_ = self._kraskov_estimator(X)

        self.nmi_: NormalizedMatrix = self.nmi(
            normalize_method=self.normalize_method,
        )

        return self

    @beartype
    def fit_transform(
        self,
        X: FloatMax2DArray,
        y: Optional[ArrayLikeFloat] = None,
    ) -> NormalizedMatrix:
        """Compute the normalized mutual information matrix and return it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features x n_dim)
            Training data.
        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        NMI : ndarray of shape (n_features, n_features)
            Pairwise normalized mutual information matrix.

        """
        self.fit(X)
        return self.nmi_

    @beartype
    def transform(
        self,
        X: Union[FloatMax2DArray, str],
    ) -> PositiveMatrix:
        """Compute the correlation/nmi distance matrix and returns it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if low_memory=True
            Training data.

        Returns
        -------
        Similarity : ndarray of shape (n_features, n_features)
            Similarity matrix.

        """
        return self.fit_transform(X)

    @beartype
    def _reset(self) -> None:
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, 'mi_'):  # noqa: WPS421
            del self.mi_  # noqa: WPS420
            del self.hxy_  # noqa: WPS420
            del self.hx_  # noqa: WPS420
            del self.hy_  # noqa: WPS420
        if hasattr(self, 'nmi_'):  # noqa: WPS421
            del self.nmi_  # noqa: WPS420

    @beartype
    def nmi(self, normalize_method: NormString) -> NormalizedMatrix:
        """Return the normalized mutual information matrix.

        Parameters
        ----------
        normalize_method : str, default='joint'
            Determines the normalization factor for the mutual information:
            - `'joint'` is the joint entropy
            - `'max'` is the maximum of the individual entropies
            - `'arithmetic'` is the mean of the individual entropies
            - `'geometric'` is the square root of the product of the individual
              entropies
            - `'min'` is the minimum of the individual entropies

        Returns
        -------
        nmi_ : ndarray of shape (n_features, n_features)
            The normalized pairwise mutual information matrix of the data.

        """
        check_is_fitted(self, attributes=['mi_', 'hxy_', 'hx_', 'hy_'])

        nmi_: np.ndarray
        if normalize_method == 'joint':
            nmi_ = self.mi_ / self.hxy_
        else:
            func: Callable = {
                'geometric': lambda arr: np.sqrt(np.prod(arr, axis=0)),
                'arithmetic': lambda arr: np.mean(arr, axis=0),
                'min': lambda arr: np.min(arr, axis=0),
                'max': lambda arr: np.max(arr, axis=0),
            }[normalize_method]
            nmi_ = self.mi_ / func([self.hx_, self.hy_])

        # ensure strict normalization within [0, 1]
        return np.clip(nmi_, a_min=0, a_max=1)

    @beartype
    def _kraskov_estimator(
        self, X: Float2DArray
    ) -> Tuple[PositiveMatrix, PositiveMatrix, PositiveMatrix, PositiveMatrix]:
        """Estimate the mutual information and entropies matrices."""
        mi: PositiveMatrix = np.empty(  # noqa: WPS317
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        hxy: PositiveMatrix = np.empty_like(mi)
        hx: PositiveMatrix = np.empty_like(mi)
        hy: PositiveMatrix = np.empty_like(mi)
        for idx_i, xi in enumerate(X.T):
            mi[idx_i, idx_i] = 1
            hxy[idx_i, idx_i] = 1
            hx[idx_i, idx_i] = 1
            hy[idx_i, idx_i] = 1
            for idx_j, xj in enumerate(X.T[idx_i + 1:], idx_i + 1):
                mi_ij, hxy_ij, hx_ij, hy_ij = kraskov_estimator(
                    xi, xj, n_neighbors=self.k,
                )
                mi[idx_i, idx_j] = mi[idx_j, idx_i] = mi_ij
                hxy[idx_i, idx_j] = hxy[idx_j, idx_i] = hxy_ij
                hx[idx_i, idx_j] = hx[idx_j, idx_i] = hx_ij
                hy[idx_i, idx_j] = hy[idx_j, idx_i] = hy_ij

        return mi, hxy, hx, hy


def kraskov_estimator(x, y, n_neighbors):
    """Compute MI(X,Y), H(X), H(Y) and H(X,Y).

    Compute mutual information, marginal and joint continuous entropies between
    the two continuous variables. This estimator is based on the KSG-Estimator
    [1]_.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples, ndims)
        Samples of two continuous random variables, must have an identical
        shape.
    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi, hx, hy, hxy : float
        Return estimates.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.

    """
    n_samples, dx = x.shape
    _, dy = y.shape
    xy = np.hstack((x, y))

    kdtree_kwargs = {'p': np.inf, 'workers': -1}

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    tree = KDTree(xy)
    radii = tree.query(
        xy, k=n_neighbors + 1, **kdtree_kwargs,
    )[0][:, 1:]  # neglect self count
    # take next smaller radii
    radii = np.nextafter(radii[:, -1], 0)

    # enforce to be strictly larger than 0
    radii = np.where(
        radii == 0,
        np.finfo(radii.dtype).resolution,
        radii,
    )

    nx, ny = [
        KDTree(z).query_ball_point(
            z, r=radii, return_length=True, **kdtree_kwargs,
        ) - 1
        for z in (x, y)
    ]

    digamma_N = digamma(n_samples)
    digamma_k = digamma(n_neighbors)
    digamma_nx = np.mean(digamma(nx + 1))
    digamma_ny = np.mean(digamma(ny + 1))

    mean_log_eps = np.mean(np.log(radii / np.mean(radii)))
    mi = digamma_N + digamma_k - digamma_nx - digamma_ny
    hxy = digamma_N - digamma_k + (dx + dy) * mean_log_eps
    hx = digamma_N - digamma_nx + dx * mean_log_eps
    hy = digamma_N - digamma_ny + dy * mean_log_eps

    return mi, hxy, hx, hy
