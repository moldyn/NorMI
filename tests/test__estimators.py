# -*- coding: utf-8 -*-
"""Tests for the estimators.

MIT License
Copyright (c) 2023, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest
from beartype.roar import BeartypeException
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from sklearn.datasets import make_moons

from nmi import NormalizedMI
from nmi import _estimators as estimators


def X1():
    """Correlated coordinates."""
    N = 1000
    return make_moons(n_samples=N, noise=0.01, random_state=69)[0]


def X1_result(method, measure):
    """Correlated coordinates results."""
    return {
        'radius': {
            'joint': 0.4516833,
            'max': 0.6152170,
            'min': 0.6295256,
            'arithmetic': 0.6222891,
            'geometric': 0.6223302,
        },
        'volume': {
            'joint': 0.4565186,
            'max': 0.6196869,
            'min': 0.6342067,
            'arithmetic': 0.6268627,
            'geometric': 0.6269047,
        },
    }[measure][method]


@pytest.mark.parametrize('invariant_measure, n_dims, radii, result, error', [
    ('radius', 1, np.arange(5), np.arange(5) / 2, None),
    ('radius', 2, np.arange(5), np.arange(5) / 2, None),
    ('volume', 1, np.arange(5), np.arange(5) / 2, None),
    (
        'volume',
        2,
        np.arange(5),
        [0, 0.40824829, 0.81649658, 1.22474487, 1.63299316],
        None,
    ),
    ('kraskov', 1, np.arange(5), np.arange(5), None),
    ('none', 1, np.arange(5), np.arange(5), BeartypeException),
])
def test__scale_nearest_neighbor_distance(
    invariant_measure, n_dims, radii, result, error,
):
    # cast radii to float to fulfill beartype typing req.
    radii = radii.astype(float)
    if error is None:
        scaled_radii = estimators._scale_nearest_neighbor_distance(
            invariant_measure, n_dims, radii,
        )
        assert_array_almost_equal(scaled_radii, result)

    else:
        with pytest.raises(error):
            estimators._scale_nearest_neighbor_distance(
                invariant_measure, n_dims, radii,
            )


@pytest.mark.parametrize('X, n_dims, error', [
    (np.random.uniform(size=(10, 9)), 1, None),
    (np.random.uniform(size=(10, 9)), 3, None),
    (np.random.uniform(size=(10, 9)), 2, ValueError),
    (np.zeros((10, 9)).astype(float), 1, ValueError),
    (np.vander((1, 2, 3, 4), 3).astype(float), 1, ValueError),
])
def test__check_X(X, n_dims, error):
    if error is None:
        estimators._check_X(X, n_dims)
    else:
        with pytest.raises(error):
            estimators._check_X(X, n_dims)


@pytest.mark.parametrize('normalize_method, X, kwargs', [
    ('joint', X1(), {}),
])
def test__reset(normalize_method, X, kwargs):
    nmi = NormalizedMI(normalize_method=normalize_method, **kwargs)
    nmi.fit(X)
    assert hasattr(nmi, 'nmi_')
    nmi._reset()
    assert not hasattr(nmi, 'nmi_')


@pytest.mark.parametrize('X, kwargs, result, error', [
    (X1(), {}, X1_result('joint', 'radius'), None),
    (
        X1(),
        {'normalize_method': 'joint', 'invariant_measure': 'radius'},
        X1_result('joint', 'radius'),
        None,
    ),
    (X1(), {'normalize_method': 'max'}, X1_result('max', 'radius'), None),
    (X1(), {'normalize_method': 'min'}, X1_result('min', 'radius'), None),
    (
        X1(),
        {'normalize_method': 'arithmetic'},
        X1_result('arithmetic', 'radius'),
        None,
    ),
    (
        X1(),
        {'normalize_method': 'geometric'},
        X1_result('geometric', 'radius'),
        None,
    ),
    (
        X1(),
        {'normalize_method': 'joint', 'invariant_measure': 'volume'},
        X1_result('joint', 'volume'),
        None,
    ),
    (
        X1(),
        {'normalize_method': 'max', 'invariant_measure': 'volume'},
        X1_result('max', 'volume'),
        None,
    ),
    (
        X1(),
        {'normalize_method': 'min', 'invariant_measure': 'volume'},
        X1_result('min', 'volume'),
        None,
    ),
    (
        X1(),
        {'normalize_method': 'arithmetic', 'invariant_measure': 'volume'},
        X1_result('arithmetic', 'volume'),
        None,
    ),
    (
        X1(),
        {'normalize_method': 'geometric', 'invariant_measure': 'volume'},
        X1_result('geometric', 'volume'),
        None,
    ),
])
def test_NormalizedMI(X, kwargs, result, error):
    # cast radii to float to fulfill beartype typing req.
    nmi = NormalizedMI(**kwargs)
    if error is None:
        nmi.fit(X)
        assert_almost_equal(nmi.nmi_[0, 1], result)
    else:
        with pytest.raises(error):
            nmi.fit(X)
