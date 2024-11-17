"""
CMB Analysis Framework

This module provides tools for analyzing CMB data, including:
- Transfer function calculations
- Power spectrum computation
- MCMC parameter estimation

The module is designed to work with various cosmological models
and provides robust numerical implementations of theoretical predictions.
"""

from .transfer import CosmicTransferFunctions
from .power_spectrum import PowerSpectrumCalculator
from .mcmc import MCMCAnalysis

__all__ = ['CosmicTransferFunctions', 'PowerSpectrumCalculator', 'MCMCAnalysis']
