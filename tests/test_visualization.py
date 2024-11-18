from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

"""
Tests for visualization components.
"""


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
    plt.close(fig)


def test_corner_plot(plotter, mock_chain, param_names):
    """Test corner plot functionality."""
    fig = plotter.plot_corner(mock_chain.reshape(-1, mock_chain.shape[2]),
                              param_names)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_chain_evolution(diagnostics, mock_chain, param_names):
    """Test MCMC chain evolution plotting."""
    fig = diagnostics.plot_chain_evolution(mock_chain, param_names)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_convergence_metrics(diagnostics, mock_chain, param_names):
    """Test convergence metrics plotting."""
    fig = diagnostics.plot_convergence_metrics(mock_chain, param_names)
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

    assert (tmp_path / "test_plot.pdf").exists()
    assert (tmp_path / "test_plot.png").exists()
    plt.close(fig)


@pytest.mark.parametrize("fig_type", [
    "power_spectra",
    "residuals",
    "corner",
    "chain_evolution",
    "convergence"
])
def test_plot_types(plotter, diagnostics, mock_cmb_data, mock_chain,
                    param_names, fig_type):
    """Test different plot types."""
    if fig_type == "power_spectra":
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
    elif fig_type == "residuals":
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
    elif fig_type == "corner":
        fig = plotter.plot_corner(
            mock_chain.reshape(-1, mock_chain.shape[2]),
            param_names
        )
    elif fig_type == "chain_evolution":
        fig = diagnostics.plot_chain_evolution(mock_chain, param_names)
    else:  # convergence
        fig = diagnostics.plot_convergence_metrics(mock_chain, param_names)

    assert isinstance(fig, Figure)
    plt.close(fig)
