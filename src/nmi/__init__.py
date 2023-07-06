# -*- coding: utf-8 -*-
"""Normalized mutual information"""
__all__ = ['NormalizedMI']

NORMS = {'joint', 'geometric', 'arithmetic', 'min', 'max'}

from ._estimators import NormalizedMI


__version__ = '0.0.0'
