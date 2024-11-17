"""
Tests for cosmological models.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from cmb_analysis.cosmology import LCDM, wCDM
from cmb_analysis.constants import constants


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


def test_lcdm_initialization():
    """Test ΛCDM model initialization."""
    model = LCDM()

    assert hasattr(model, 'param_info')
    assert 'H0' in model.param_info
    assert 'omega_b' in model.param_info
    assert 'omega_cdm' in model.param_info


def test_wcdm_initialization():
    """Test wCDM model initialization."""
    model = wCDM()

    assert hasattr(model, 'param_info')
    assert 'w' in model.param_info
    assert model.param_info['w'][0] == -1.0  # Default w value


@pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0])
def test_hubble_parameter(z, fiducial_params):
    """Test Hubble parameter computation."""
    model = LCDM()
    H = model.H(z, fiducial_params)

    # Basic sanity checks
    assert H > 0
    # H(z) should be larger than H0 for z > 0
    assert H >= fiducial_params['H0']


def test_hubble_parameter_array():
    """Test Hubble parameter computation with array input."""
    model = LCDM()
    z = np.linspace(0, 2, 100)
    H = model.H(z, fiducial_params)

    assert len(H) == len(z)
    assert np.all(H > 0)
    assert np.all(np.diff(H) > 0)  # H(z) should be monotonically increasing


def test_angular_diameter_distance():
    """Test angular diameter distance computation."""
    model = LCDM()
    z = 1089.8  # Last scattering surface
    D_A = model.angular_diameter_distance(z, fiducial_params)

    # Compare with Planck 2018 value
    planck_value = 13.8688e3  # Mpc
    assert_allclose(D_A, planck_value, rtol=0.01)


def test_lcdm_wcdm_consistency():
    """Test that wCDM reduces to ΛCDM when w = -1."""
    z = np.linspace(0, 2, 100)

    lcdm = LCDM()
    wcdm = wCDM()

    # Add w = -1 to parameters
    wcdm_params = fiducial_params.copy()
    wcdm_params['w'] = -1.0

    H_lcdm = lcdm.H(z, fiducial_params)
    H_wcdm = wcdm.H(z, wcdm_params)

    assert_array_almost_equal(H_lcdm, H_wcdm)


def test_invalid_parameters():
    """Test handling of invalid parameters."""
    model = LCDM()

    invalid_params = fiducial_params.copy()
    invalid_params['H0'] = -70  # Invalid negative Hubble constant

    with pytest.raises(ValueError):
        model.validate_parameters(invalid_params)


def test_omega_m_computation():
    """Test total matter density computation."""
    model = LCDM()
    omega_m = fiducial_params['omega_b'] + fiducial_params['omega_cdm']
    h2 = (fiducial_params['H0']/100)**2
    expected_Omega_m = omega_m / h2

    computed_Omega_m = model.omega_m(fiducial_params)
    assert_allclose(computed_Omega_m, expected_Omega_m)


def test_age_of_universe():
    """Test age of universe computation."""
    model = LCDM()
    age = model.age_of_universe(fiducial_params)

    # Compare with Planck 2018 value
    planck_age = 13.797  # Gyr
    assert_allclose(age, planck_age, rtol=0.01)


@pytest.mark.parametrize("cosmology,params", [
    (LCDM, {'H0': 70, 'omega_b': 0.0486, 'omega_cdm': 0.2589}),
    (wCDM, {'H0': 70, 'omega_b': 0.0486, 'omega_cdm': 0.2589, 'w': -1.0}),
])
def test_physical_constraints(cosmology, params):
    """Test physical constraints and consistency relations."""
    model = cosmology()
    z = np.linspace(0, 1000, 1000)

    # Test Hubble parameter evolution
    H = model.H(z, params)
    assert np.all(np.diff(H) > 0)  # H(z) should increase with z

    # Test angular diameter distance
    D_A = np.array([model.angular_diameter_distance(zi, params)
                   for zi in z[1:]])
    # D_A should decrease after reaching maximum
    assert np.all(np.diff(D_A) < 0)

    # Test luminosity distance
    D_L = np.array([model.luminosity_distance(zi, params) for zi in z[1:]])
    assert np.all(np.diff(D_L) > 0)  # D_L should increase monotonically


def test_error_handling():
    """Test error handling in cosmological computations."""
    model = LCDM()

    # Test with missing parameters
    with pytest.raises(ValueError):
        model.H(1.0, {'H0': 70})

    # Test with invalid redshift
    with pytest.raises(ValueError):
        model.angular_diameter_distance(-1, fiducial_params)
