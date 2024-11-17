"""
Tests for visualization components.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cmb_analysis.visualization import CMBPlotter, MCMCDiagnostics


@pytest.fixture
def plotter():
    """Initialize CMB plotter."""
    return CMBPlotter()


@pytest.fixture
def diagnostics():
    """Initialize MCMC diagnostics."""
    return MCMCDiagnostics()


def test_plotter_initialization(plotter):
    """Test plotter initialization."""
    assert hasattr(plotter, 'colors')
    assert hasattr(plotter, 'fig_size')
    plt.close('all')


def test_power_spectra_plot(plotter, mock_cmb_data):
    """Test power spectra plotting functionality."""
    theory = {
        'cl_tt': mock_cmb_data['cl_tt'],
        'cl_te': mock_cmb_data['cl_te'],
        'cl_ee': mock_cmb_data['cl_ee']
    }
    data = theory.copy()
    errors = {
        'cl_tt': mock_cmb_data['noise_tt'],
        'cl_te': mock_cmb_data['noise_te'],
        'cl_ee': mock_cmb_data['noise_ee']
    }

    fig = plotter.plot_power_spectra(theory, data, errors)
    assert isinstance(fig, Figure)

    # Check plot components
    assert len(fig.axes) == 3  # Should have 3 subplots
    for ax in fig.axes:
        assert len(ax.lines) > 0  # Should have plotted lines
        assert ax.get_xlabel()  # Should have labels
        assert ax.get_ylabel()

    plt.close(fig)


def test_residuals_plot(plotter, mock_cmb_data):
    """Test residuals plotting functionality."""
    theory = {
        'cl_tt': mock_cmb_data['cl_tt'],
        'cl_te': mock_cmb_data['cl_te'],
        'cl_ee': mock_cmb_data['cl_ee']
    }
    data = theory.copy()
    errors = {
        'cl_tt': mock_cmb_data['noise_tt'],
        'cl_te': mock_cmb_data['noise_te'],
        'cl_ee': mock_cmb_data['noise_ee']
    }

    fig = plotter.plot_residuals(theory, data, errors)
    assert isinstance(fig, Figure)

    # Check residuals properties
    for ax in fig.axes:
        # Should have zero line
        assert any(line.get_linestyle() == '--' for line in ax.lines)
        # Should have y-label containing "Residuals"
        assert "Residuals" in ax.get_ylabel()

    plt.close(fig)


def test_corner_plot(plotter):
    """Test corner plot functionality."""
    # Create mock MCMC samples
    n_samples = 1000
    n_params = 3
    samples = np.random.normal(size=(n_samples, n_params))
    labels = ['param1', 'param2', 'param3']

    fig = plotter.plot_corner(samples, labels)
    assert isinstance(fig, Figure)

    # Check corner plot properties
    assert len(fig.axes) == n_params * n_params
    plt.close(fig)


def test_chain_evolution(diagnostics):
    """Test MCMC chain evolution plotting."""
    # Create mock chain
    n_steps = 1000
    n_walkers = 32
    n_params = 3
    chain = np.random.normal(size=(n_steps, n_walkers, n_params))
    param_names = ['param1', 'param2', 'param3']

    fig = diagnostics.plot_chain_evolution(chain, param_names)
    assert isinstance(fig, Figure)

    # Check chain plot properties
    assert len(fig.axes) == n_params
    plt.close(fig)


def test_autocorrelation_plot(diagnostics):
    """Test autocorrelation plotting functionality."""
    # Create mock chain
    n_steps = 1000
    n_walkers = 32
    n_params = 3
    chain = np.random.normal(size=(n_steps, n_walkers, n_params))
    param_names = ['param1', 'param2', 'param3']

    fig = diagnostics.plot_autocorrelation(chain, param_names)
    assert isinstance(fig, Figure)

    # Check autocorrelation plot properties
    assert len(fig.axes) == n_params
    plt.close(fig)


def test_convergence_metrics(diagnostics):
    """Test convergence metrics plotting."""
    # Create mock chain
    n_steps = 1000
    n_walkers = 32
    n_params = 3
    chain = np.random.normal(size=(n_steps, n_walkers, n_params))
    param_names = ['param1', 'param2', 'param3']

    fig = diagnostics.plot_convergence_metrics(chain, param_names)
    assert isinstance(fig, Figure)

    # Check metrics plot properties
    assert len(fig.axes) == 2  # Should have GR stats and effective sample size
    plt.close(fig)


def test_diagnostic_summary(diagnostics):
    """Test comprehensive diagnostic summary plot."""
    # Create mock chain and data
    n_steps = 1000
    n_walkers = 32
    n_params = 3
    chain = np.random.normal(size=(n_steps, n_walkers, n_params))
    param_names = ['param1', 'param2', 'param3']
    acceptance_fraction = 0.5

    fig = diagnostics.plot_diagnostic_summary(
        chain, param_names, acceptance_fraction
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_save_publication_plots(plotter, mock_cmb_data, tmp_path):
    """Test saving publication-quality plots."""
    theory = {
        'cl_tt': mock_cmb_data['cl_tt'],
        'cl_te': mock_cmb_data['cl_te'],
        'cl_ee': mock_cmb_data['cl_ee']
    }
    data = theory.copy()
    errors = {
        'cl_tt': mock_cmb_data['noise_tt'],
        'cl_te': mock_cmb_data['noise_te'],
        'cl_ee': mock_cmb_data['noise_ee']
    }

    fig = plotter.plot_power_spectra(theory, data, errors)
    filename = tmp_path / "test_plot"
    plotter.save_publication_plots(filename, fig)

    # Check that files were created
    assert (tmp_path / "test_plot.pdf").exists()
    assert (tmp_path / "test_plot.png").exists()
    plt.close(fig)


def test_style_consistency(plotter):
    """Test plot style consistency."""
    # Create simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    # Check style properties
    assert plt.rcParams['font.family'] == ['serif']
    assert plt.rcParams['axes.grid'] == True
    plt.close(fig)


@pytest.mark.parametrize("fig_type", [
    "power_spectra",
    "residuals",
    "corner",
    "chain_evolution",
    "convergence"
])
def test_plot_types(plotter, diagnostics, mock_cmb_data, fig_type):
    """Test different plot types."""
    if fig_type == "power_spectra":
        fig = plotter.plot_power_spectra(
            {'cl_tt': mock_cmb_data['cl_tt'],
             'cl_te': mock_cmb_data['cl_te'],
             'cl_ee': mock_cmb_data['cl_ee']},
            {'cl_tt': mock_cmb_data['cl_tt'],
             'cl_te': mock_cmb_data['cl_te'],
             'cl_ee': mock_cmb_data['cl_ee']},
            {'cl_tt': mock_cmb_data['noise_tt'],
             'cl_te': mock_cmb_data['noise_te'],
             'cl_ee': mock_cmb_data['noise_ee']}
        )
    elif fig_type == "residuals":
        fig = plotter.plot_residuals(
            {'cl_tt': mock_cmb_data['cl_tt'],
             'cl_te': mock_cmb_data['cl_te'],
             'cl_ee': mock_cmb_data['cl_ee']},
            {'cl_tt': mock_cmb_data['cl_tt'],
             'cl_te': mock_cmb_data['cl_te'],
             'cl_ee': mock_cmb_data['cl_ee']},
            {'cl_tt': mock_cmb_data['noise_tt'],
             'cl_te': mock_cmb_data['noise_te'],
             'cl_ee': mock_cmb_data['noise_ee']}
        )
    elif fig_type == "corner":
        samples = np.random.normal(size=(1000, 3))
        fig = plotter.plot_corner(samples, ['param1', 'param2', 'param3'])
    elif fig_type == "chain_evolution":
        chain = np.random.normal(size=(1000, 32, 3))
        fig = diagnostics.plot_chain_evolution(
            chain, ['param1', 'param2', 'param3']
        )
    elif fig_type == "convergence":
        chain = np.random.normal(size=(1000, 32, 3))
        fig = diagnostics.plot_convergence_metrics(
            chain, ['param1', 'param2', 'param3']
        )

    assert isinstance(fig, Figure)
    plt.close(fig)
