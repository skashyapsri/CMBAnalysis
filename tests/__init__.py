"""
Test suite for CMB Analysis package.

This module contains common utilities and fixtures used across tests.
"""

import numpy as np
import pytest
from pathlib import Path

# Define test data directory
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def mock_cmb_data():
    """Generate mock CMB data for testing."""
    # Generate ell values
    ell = np.arange(2, 2500)

    # Generate mock TT spectrum
    cl_tt = 1000 * (ell/100)**(-2) * np.exp(-(ell/1000)**2)

    # Generate mock TE spectrum
    cl_te = 0.4 * cl_tt * np.sin(np.pi * ell/180)

    # Generate mock EE spectrum
    cl_ee = 0.1 * cl_tt

    # Add noise
    noise_tt = 0.01 * np.sqrt(cl_tt)
    noise_te = 0.02 * np.sqrt(np.abs(cl_te))
    noise_ee = 0.02 * np.sqrt(cl_ee)

    return {
        'ell': ell,
        'cl_tt': cl_tt,
        'cl_te': cl_te,
        'cl_ee': cl_ee,
        'noise_tt': noise_tt,
        'noise_te': noise_te,
        'noise_ee': noise_ee
    }


@pytest.fixture
def fiducial_params():
    """Return fiducial cosmological parameters."""
    return {
        'H0': 67.32,
        'omega_b': 0.02237,
        'omega_cdm': 0.1200,
        'tau': 0.0544,
        'ns': 0.9649,
        'ln10As': 3.044
    }


@pytest.fixture
def tolerance():
    """Return numerical tolerances for tests."""
    return {
        'rtol': 1e-7,  # Relative tolerance
        'atol': 1e-7   # Absolute tolerance
    }
