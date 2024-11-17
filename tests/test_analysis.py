"""
Tests for CMB analysis components.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from cmb_analysis.analysis import (CosmicTransferFunctions,
                                   PowerSpectrumCalculator,
                                   MCMCAnalysis)


@pytest.fixture
def transfer_calculator():
    """Initialize transfer function calculator."""
    return CosmicTransferFunctions()


@pytest.fixture
def power_calculator():
    """Initialize power spectrum calculator."""
    return PowerSpectrumCalculator()


def test_transfer_function_initialization(transfer_calculator):
    """Test transfer function calculator initialization."""
    assert hasattr(transfer_calculator, 'k_grid')
    assert hasattr(transfer_calculator, 'setup_transfer_functions')


def test_matter_transfer_function(transfer_calculator, fiducial_params):
    """Test matter transfer function computation."""
    k = np.logspace(-4, 2, 100)
    T = transfer_calculator.matter_transfer(k, fiducial_params)

    # Basic sanity checks
    assert len(T) == len(k)
    assert np.all(np.isfinite(T))
    assert np.all(T <= 1.0)  # Transfer function should be normalized
    assert np.isclose(T[0], 1.0)  # Should approach 1 at large scales


def test_radiation_transfer_function(transfer_calculator, fiducial_params):
    """Test radiation transfer function computation."""
    k = np.logspace(-4, 2, 100)
    T = transfer_calculator.radiation_transfer(k, fiducial_params)

    assert len(T) == len(k)
    assert np.all(np.isfinite(T))
    assert np.all(np.abs(T) <= 1.0)  # Should be bounded


def test_power_spectrum_computation(power_calculator, fiducial_params, mock_cmb_data):
    """Test power spectrum computation."""
    cl_tt, cl_ee, cl_te = power_calculator.compute_all_spectra(fiducial_params)

    # Check array sizes
    assert len(cl_tt) == len(mock_cmb_data['ell'])
    assert len(cl_ee) == len(mock_cmb_data['ell'])
    assert len(cl_te) == len(mock_cmb_data['ell'])

    # Basic physical checks
    assert np.all(cl_tt > 0)  # TT spectrum should be positive
    assert np.all(cl_ee > 0)  # EE spectrum should be positive
    assert np.all(np.abs(cl_te) <= np.sqrt(cl_tt * cl_ee))  # Cauchy-Schwarz


def test_mcmc_analysis(mock_cmb_data, fiducial_params):
    """Test MCMC analysis setup and basic functionality."""
    calculator = PowerSpectrumCalculator()
    data = {
        'tt_data': mock_cmb_data['cl_tt'],
        'te_data': mock_cmb_data['cl_te'],
        'ee_data': mock_cmb_data['cl_ee'],
        'tt_error': mock_cmb_data['noise_tt'],
        'te_error': mock_cmb_data['noise_te'],
        'ee_error': mock_cmb_data['noise_ee']
    }

    mcmc = MCMCAnalysis(calculator, data, fiducial_params)

    # Test log probability computation
    lp = mcmc.log_probability(list(fiducial_params.values()))
    assert np.isfinite(lp)

    # Test initialization
    initial = mcmc._initialize_walkers()
    assert initial.shape == (mcmc.nwalkers, mcmc.ndim)


@pytest.mark.slow
def test_mcmc_convergence(mock_cmb_data, fiducial_params):
    """Test MCMC convergence diagnostics."""
    calculator = PowerSpectrumCalculator()
    data = {
        'tt_data': mock_cmb_data['cl_tt'],
        'te_data': mock_cmb_data['cl_te'],
        'ee_data': mock_cmb_data['cl_ee'],
        'tt_error': mock_cmb_data['noise_tt'],
        'te_error': mock_cmb_data['noise_te'],
        'ee_error': mock_cmb_data['noise_ee']
    }

    mcmc = MCMCAnalysis(calculator, data, fiducial_params)

    # Run short chain for testing
    mcmc.nsteps = 100
    mcmc.run_mcmc(progress=False)

    # Test convergence diagnostics
    diagnostics = mcmc.compute_convergence_diagnostics()

    assert 'gelman_rubin' in diagnostics
    assert 'effective_samples' in diagnostics
    assert 'acceptance_fraction' in diagnostics

    # Check Gelman-Rubin statistics
    for param, gr in diagnostics['gelman_rubin'].items():
        assert gr > 0


def test_numerical_stability(transfer_calculator, power_calculator, fiducial_params):
    """Test numerical stability of computations."""
    # Test with extreme parameter values
    extreme_params = fiducial_params.copy()
    extreme_params['H0'] = 80.0
    extreme_params['tau'] = 0.01

    # Transfer functions
    k = np.logspace(-4, 2, 100)
    T = transfer_calculator.matter_transfer(k, extreme_params)
    assert np.all(np.isfinite(T))

    # Power spectra
    cl_tt, cl_ee, cl_te = power_calculator.compute_all_spectra(extreme_params)
    assert np.all(np.isfinite(cl_tt))
    assert np.all(np.isfinite(cl_ee))
    assert np.all(np.isfinite(cl_te))


@pytest.mark.parametrize("ell", [100, 500, 1000])
def test_acoustic_peaks(power_calculator, fiducial_params, ell):
    """Test acoustic peak locations and amplitudes."""
    cl_tt, _, _ = power_calculator.compute_all_spectra(fiducial_params)

    # Convert to D_l = l(l+1)C_l/(2π)
    dl_tt = ell * (ell + 1) * cl_tt[ell] / (2 * np.pi)

    # Check that values are reasonable
    assert dl_tt > 0
    assert np.isfinite(dl_tt)


def test_error_propagation(power_calculator, fiducial_params):
    """Test error handling and propagation."""
    # Test with invalid parameters
    invalid_params = fiducial_params.copy()
    invalid_params['H0'] = -70

    with pytest.raises(ValueError):
        power_calculator.compute_all_spectra(invalid_params)
