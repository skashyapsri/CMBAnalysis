from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
import numpy as np
"""
Tests for analysis components.
"""


def test_matter_transfer_function(transfer_calculator, fiducial_params):
    """Test matter transfer function computation."""
    k = np.logspace(-4, 2, 100)
    T = transfer_calculator.matter_transfer(k, fiducial_params)

    # Basic sanity checks
    assert len(T) == len(k)
    assert np.all(np.isfinite(T))
    assert np.all(np.abs(T) <= 1.0)  # Transfer function should be normalized
    assert np.isclose(T[0], 1.0, rtol=1e-2)  # Should approach 1 at large scales


def test_power_spectrum_computation(power_calculator, fiducial_params, mock_cmb_data):
    """Test power spectrum computation."""
    # Compute spectra
    cl_tt, cl_ee, cl_te = power_calculator.compute_all_spectra(fiducial_params)

    # Check array sizes
    assert len(cl_tt) == len(mock_cmb_data['ell'])
    assert len(cl_ee) == len(mock_cmb_data['ell'])
    assert len(cl_te) == len(mock_cmb_data['ell'])

    # Check physical constraints
    assert np.all(np.isfinite(cl_tt))
    assert np.all(cl_tt >= 0)  # TT spectrum should be positive
    assert np.all(cl_ee >= 0)  # EE spectrum should be positive
    assert np.all(np.abs(cl_te) <= np.sqrt(cl_tt * cl_ee))  # Cauchy-Schwarz


def test_mcmc_analysis(mock_cmb_data, fiducial_params):
    """Test basic MCMC setup."""
    from cmb_analysis.analysis import MCMCAnalysis

    analyzer = MCMCAnalysis(mock_cmb_data, fiducial_params)
    lp = analyzer.log_probability(list(fiducial_params.values()))

    assert np.isfinite(lp)
    assert isinstance(lp, float)


@pytest.mark.slow
def test_mcmc_convergence(mock_cmb_data, fiducial_params):
    """Test MCMC convergence diagnostics."""
    from cmb_analysis.analysis import MCMCAnalysis

    analyzer = MCMCAnalysis(mock_cmb_data, fiducial_params)
    analyzer.nsteps = 100  # Reduce steps for testing

    chain = analyzer.run_mcmc()
    stats = analyzer.compute_convergence_diagnostics()

    assert 'gelman_rubin' in stats
    assert 'effective_samples' in stats
    assert 'acceptance_fraction' in stats
    assert 0 <= stats['acceptance_fraction'] <= 1


def test_numerical_stability(transfer_calculator, power_calculator, fiducial_params):
    """Test numerical stability of computations."""
    # Test with extreme parameter values
    extreme_params = fiducial_params.copy()
    extreme_params['H0'] = 80.0
    extreme_params['tau'] = 0.01

    k = np.logspace(-4, 2, 100)
    T = transfer_calculator.matter_transfer(k, extreme_params)
    assert np.all(np.isfinite(T))

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

    assert dl_tt > 0
    assert np.isfinite(dl_tt)


def test_error_propagation(power_calculator, fiducial_params):
    """Test error handling and propagation."""
    # Test with invalid parameters
    invalid_params = fiducial_params.copy()
    invalid_params['H0'] = -70  # Invalid negative value

    with pytest.raises(ValueError):
        power_calculator.compute_all_spectra(invalid_params)
