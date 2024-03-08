# -*- coding: utf-8 -*-
"""Class for estimating normalized mutual information.

MIT License
Copyright (c) 2023, Daniel Nagel
All rights reserved.

"""
__all__ = ['NormalizedMI']  # noqa: WPS410

import numpy as np
from beartype import beartype
from beartype.typing import Callable, List, Optional, Tuple, Union
from scipy.spatial import KDTree
from scipy.special import digamma
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from normi._typing import (  # noqa: WPS436
    ArrayLikeFloat,
    Float,
    Float2DArray,
    FloatArray,
    FloatMatrix,
    FloatMax2DArray,
    Int,
    InvMeasureString,
    NormalizedMatrix,
    NormString,
    PositiveFloat,
    PositiveInt,
    PositiveMatrix,
)


class NormalizedMI(BaseEstimator):
    r"""Class for estimating the normalized mutual information.

    Parameters
    ----------
    n_dims : int, default=1
        Dimensionality of input vectors.
    normalize_method : str, default='geometric'
        Determines the normalization factor for the mutual information:<br/>
        - `'joint'` is the joint entropy<br/>
        - `'max'` is the maximum of the individual entropies<br/>
        - `'arithmetic'` is the mean of the individual entropies<br/>
        - `'geometric'` is the square root of the product of the individual
          entropies<br/>
        - `'min'` is the minimum of the individual entropies
    invariant_measure : str, default='volume'
        - `'radius'` normalizing by mean k-nn radius<br/>
        - `'volume'` normalizing by mean k-nn volume<br/>
        - `'kraskov'` no normalization
    k : int, default=5
        Number of nearest neighbors to use in $k$-nn estimator.
    n_jobs; int, default=-1
        Number of jobs to use, `-1` uses as many as cores are available.
    verbose : bool, default=True
        Setting verbose mode.

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
    >>> from normi import NormalizedMI
    >>> x = np.linspace(0, np.pi, 1000)
    >>> data = np.array([np.cos(x), np.cos(x + np.pi / 6)]).T
    >>> nmi = NormalizedMI()
    >>> nmi.fit(data)
    NormalizedMI()
    >>> nmi.nmi_
    array([[1.        , 0.79868365],
           [0.79868365, 1.        ]])

    """

    _dtype: np.dtype = np.float64

    @beartype
    def __init__(
        self,
        *,
        n_dims: PositiveInt = 1,
        normalize_method: NormString = 'geometric',
        invariant_measure: InvMeasureString = 'volume',
        k: PositiveInt = 5,
        n_jobs: Int = -1,
        verbose: bool = True,
    ):
        """Initialize NormalizedMI class."""
        self.n_dims: PositiveInt = n_dims
        self.normalize_method: NormString = normalize_method
        self.invariant_measure: InvMeasureString = invariant_measure
        self.k: PositiveInt = k
        self.verbose: bool = verbose
        self.n_jobs: Int = n_jobs

    @beartype
    def fit(
        self,
        X: FloatMax2DArray,
        y: Optional[ArrayLikeFloat] = None,
    ):
        """Compute the normalized mutual information matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features x n_dims)
            Training data.
        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        self._reset()

        _check_X(X=X, n_dims=self.n_dims)

        # define number of features and samples
        n_samples: int
        n_cols: int
        n_samples, n_cols = X.shape
        self._n_samples: int = n_samples
        self._n_features: int = n_cols // self.n_dims

        # scale input
        X = StandardScaler().fit_transform(X)
        X = np.split(X, self._n_features, axis=1)

        self.mi_: PositiveMatrix
        self.hxy_: FloatMatrix
        self.hx_: FloatMatrix
        self.hy_: FloatMatrix

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
        X : ndarray of shape (n_samples, n_features x n_dims)
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
    def nmi(
        self, normalize_method: Optional[NormString] = None,
    ) -> NormalizedMatrix:
        """Return the normalized mutual information matrix.

        Parameters
        ----------
        normalize_method : str, default=None
            If `None` use class definition.
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

        if normalize_method is None:
            normalize_method = self.normalize_method

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
        self, X: List[Float2DArray],
    ) -> Tuple[PositiveMatrix, FloatMatrix, FloatMatrix, FloatMatrix]:
        """Estimate the mutual information and entropies matrices."""
        mi: PositiveMatrix = np.empty(  # noqa: WPS317
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        hxy: FloatMatrix = np.empty_like(mi)
        hx: FloatMatrix = np.empty_like(mi)
        hy: FloatMatrix = np.empty_like(mi)

        pb = tqdm(
            total=int(self._n_features * (self._n_features - 1) / 2),
            disable=not self.verbose,
            desc='NMI Estimation',
        )

        for idx_i, xi in enumerate(X):
            mi[idx_i, idx_i] = 1
            hxy[idx_i, idx_i] = 1
            hx[idx_i, idx_i] = 1
            hy[idx_i, idx_i] = 1
            for idx_j, xj in enumerate(X[idx_i + 1:], idx_i + 1):
                mi_ij, hxy_ij, hx_ij, hy_ij = kraskov_estimator(
                    xi,
                    xj,
                    n_neighbors=self.k,
                    invariant_measure=self.invariant_measure,
                    n_jobs=self.n_jobs,
                )
                mi[idx_i, idx_j] = mi[idx_j, idx_i] = mi_ij
                hxy[idx_i, idx_j] = hxy[idx_j, idx_i] = hxy_ij
                hx[idx_i, idx_j] = hx[idx_j, idx_i] = hx_ij
                hy[idx_i, idx_j] = hy[idx_j, idx_i] = hy_ij

                pb.update()

        return mi, hxy, hx, hy


@beartype
def _scale_nearest_neighbor_distance(
    invariant_measure: InvMeasureString,
    n_dims: PositiveInt,
    radii: FloatArray,
) -> FloatArray:
    """Apply invariant measure rescaling to radii.

    Parameters
    ----------
    invariant_measure : str, default='radius'
        - `'radius'` normalizing by mean k-nn radius<br/>
        - `'volume'` normalizing by mean k-nn volume<br/>
        - `'kraskov'` no normalization
    n_dims : int
        Dimensionality of the embedding space used to estimate the radii.
    radii : ndarray, shape (n_samples, )
        $k$-NN radii of each sample.

    Returns
    -------
    mi, hx, hy, hxy : float
        Return estimates.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.

    """
    if invariant_measure == 'radius':
        return radii / np.mean(radii)
    elif invariant_measure == 'volume':
        return radii / (
            np.mean(radii ** n_dims) ** (1 / n_dims)
        )
    elif invariant_measure == 'kraskov':
        return radii
    # This should never be reached
    raise NotImplementedError(  # no cover
        f'Selected invariant measure {invariant_measure} is not implemented.',
    )


@beartype
def kraskov_estimator(
    x: Float2DArray,
    y: Float2DArray,
    n_neighbors: PositiveInt,
    invariant_measure: InvMeasureString,
    n_jobs: Int,
) -> Tuple[PositiveFloat, Float, Float, Float]:
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
    invariant_measure : str, default='radius'
        - `'radius'` normalizing by mean k-nn radius<br/>
        - `'volume'` normalizing by mean k-nn volume<br/>
        - `'kraskov'` no normalization
    n_jobs : int
        Number of jobs to use.

    Returns
    -------
    mi, hx, hy, hxy : float
        Return estimates.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.

    """
    n_samples: int
    dx: int
    dy: int
    n_samples, dx = x.shape
    _, dy = y.shape
    xy = np.hstack((x, y))

    kdtree_kwargs = {'p': np.inf, 'workers': n_jobs}

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    tree = KDTree(xy)
    radii: FloatArray = tree.query(
        xy, k=n_neighbors + 1, **kdtree_kwargs,
    )[0][:, 1:]  # neglect self count
    # take next smaller radii
    radii: FloatArray = np.nextafter(radii[:, -1], 0)

    # enforce to be strictly larger than 0
    radii: FloatArray = np.where(
        radii == 0,
        np.finfo(radii.dtype).resolution,
        radii,
    )

    nx: FloatArray
    ny: FloatArray
    nx, ny = [
        KDTree(z).query_ball_point(
            z, r=radii, return_length=True, **kdtree_kwargs,
        ) - 1  # fix self count
        for z in (x, y)
    ]

    # scale radiis
    radii = _scale_nearest_neighbor_distance(
        invariant_measure=invariant_measure,
        n_dims=(dx + dy),
        radii=radii,
    )

    digamma_N: Float = digamma(n_samples)
    digamma_k: Float = digamma(n_neighbors)
    digamma_nx: Float = np.mean(digamma(nx + 1))
    digamma_ny: Float = np.mean(digamma(ny + 1))
    mean_log_eps: Float = np.mean(np.log(radii))

    return (
        digamma_N + digamma_k - digamma_nx - digamma_ny,  # mi
        digamma_N - digamma_k + (dx + dy) * mean_log_eps,  # hxy
        digamma_N - digamma_nx + dx * mean_log_eps,  # hx
        digamma_N - digamma_ny + dy * mean_log_eps,  # hy
    )


@beartype
def _check_X(X: Float2DArray, n_dims: PositiveInt):
    """Sanity check of the input to ensure correct format and dimension."""
    # parse data
    if X.shape[1] < 2 * n_dims:
        raise ValueError('At least two variables need to be provided')
    stds = np.std(X, axis=0)
    invalid_stds = (stds == 0) | (np.isnan(stds))
    if np.any(invalid_stds):
        idxs = np.where(invalid_stds)[0]
        raise ValueError(
            f'Columns {idxs} have a standard deviation of zero or NaN. '
            'These columns cannot be used for estimating the NMI.',
        )

    _, n_cols = X.shape
    n_features = n_cols // n_dims
    if n_cols != n_features * n_dims:
        raise ValueError(
            'The number of provided columns needs to be a multiple of the '
            'specified dimensionality `n_dims`.',
        )
