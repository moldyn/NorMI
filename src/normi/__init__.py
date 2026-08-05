# -*- coding: utf-8 -*-
"""Normalized mutual information"""

__all__ = ['NormalizedMI']

NORMS = {'joint', 'geometric', 'arithmetic', 'min', 'max'}
INVMEASURES = {
    'radius',
    'radius_median',
    'volume',
    'volume_median',
    'volume_stable',
    'kraskov',
}

from ._estimators import NormalizedMI


__version__ = '0.2.1'
