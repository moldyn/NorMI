# -*- coding: utf-8 -*-
"""Tests for the utils module.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from nmi import _utils

# Current directory
HERE = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(HERE, 'test_files')


@pytest.mark.parametrize(
    'arr, fmt, kwargs',
    (
        (np.random.normal(size=(10, 50)), '%.5f', {}),
        (np.random.normal(size=(10, 50)), '%.10f', {}),
        (np.random.normal(size=(50)), '%.10f', {}),
        (np.random.normal(size=(50)), '%.10f', {'header': 'header'}),
    ),
)
def test_save_load_clusters(arr, fmt, kwargs, tmpdir):
    """Test save/load file."""
    # save and load clusters
    filename = str(tmpdir.mkdir('sub').join('savetxt_test'))
    _utils.savetxt(filename, arr, fmt, **kwargs)

    arr_loaded = np.loadtxt(filename)
    assert_array_almost_equal(arr_loaded, arr, decimal=5)
