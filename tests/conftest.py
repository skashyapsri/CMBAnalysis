from cmb_analysis.visualization import CMBPlotter, MCMCDiagnostics
from cmb_analysis.analysis import PowerSpectrumCalculator, MCMCAnalysis
from cmb_analysis.cosmology import LCDM, wCDM
import matplotlib
import numpy as np
import pytest

"""
Common test fixtures for CMB analysis.
"""

matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif'
})


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def mock_cmb_data(fiducial_params):
    """Generate mock CMB data for testing."""
    calculator = PowerSpectrumCalculator()

    # Generate spectra
    ell = np.arange(2, 2500)
    cl_tt, cl_ee, cl_te = calculator.compute_all_spectra(fiducial_params)

    # Add noise
    noise_level = 0.01
    noise_tt = noise_level * np.sqrt(np.abs(cl_tt))
    noise_te = noise_level * np.sqrt(np.abs(cl_te))
    noise_ee = noise_level * np.sqrt(np.abs(cl_ee))

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
def transfer_calculator():
    """Initialize transfer function calculator for testing."""
    from cmb_analysis.analysis.transfer import CosmicTransferFunctions
    calculator = CosmicTransferFunctions()
    calculator.setup_transfer_functions()  # Ensure initialization
    return calculator


@pytest.fixture
def power_calculator():
    """Initialize power spectrum calculator for testing."""
    calculator = PowerSpectrumCalculator()
    calculator.setup_integration()  # Ensure initialization
    return calculator


@pytest.fixture
def plotter():
    """Initialize CMB plotter for testing."""
    return CMBPlotter()


@pytest.fixture
def diagnostics():
    """Initialize MCMC diagnostics for testing."""
    return MCMCDiagnostics()


@pytest.fixture
def mock_chain(fiducial_params):
    """Generate mock MCMC chain for testing."""
    n_steps = 1000
    n_walkers = 32
    n_params = len(fiducial_params)

    # Generate chain around fiducial values
    params = np.array(list(fiducial_params.values()))
    chain = np.random.normal(
        loc=params,
        scale=np.abs(params) * 0.01,
        size=(n_steps, n_walkers, n_params)
    )
    return chain


@pytest.fixture
def param_names(fiducial_params):
    """Return parameter names for testing."""
    return list(fiducial_params.keys())
