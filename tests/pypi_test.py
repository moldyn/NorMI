"""Simple test for PyPI."""

import numpy as np

from normi import NormalizedMI


if __name__ == '__main__':
    x = np.linspace(0, np.pi, 1000)
    data = np.array([np.cos(x), np.cos(x + np.pi / 6)]).T

    nmi = NormalizedMI(verbose=False)
    nmi.fit(data)
    assert nmi.nmi_.shape == (2, 2)
