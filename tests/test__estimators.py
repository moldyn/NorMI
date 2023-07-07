# -*- coding: utf-8 -*-
"""Tests for the estimators.

MIT License
Copyright (c) 2023, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest
from beartype.roar import BeartypeException

from nmi import NormalizedMI
from nmi import _estimators as estimators


def X1():
    """Correlated coordinates."""
    N = 10000
    return np.random.normal(
        loc=[
            3 * np.heaviside(np.arange(N) - N // 2, 0)
            for _ in range(2)
        ],
        size=(2, N),
    ).T


def X1_result(method, measure):
    """Correlated coordinates results."""
    return {
        ('joint', 'radius'): 1,
        ('joint', 'volume'): 1,
        ('joint', 'kraskov'): 1,
        ('max', 'radius'): 1,
        ('max', 'volume'): 1,
        ('min', 'radius'): 1,
        ('arithmetic', 'radius'): 1,
        ('arithmetic', 'volume'): 1,
        ('geometric', 'radius'): 1,
        ('geometric', 'volume'): 1,
    }[(method, measure)]


@pytest.mark.parametrize('X, n_dims, error', [
    (np.random.uniform(size=(10, 9)), 1, None),
    (np.random.uniform(size=(10, 9)), 3, None),
    (np.random.uniform(size=(10, 9)), 2, ValueError),
    (np.zeros((10, 9)), 1, ValueError),
    (np.vander((1, 2, 3, 4), 3), 1, ValueError),
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
