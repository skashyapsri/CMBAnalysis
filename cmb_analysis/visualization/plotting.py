"""
Comprehensive plotting utilities for CMB analysis.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import corner
from scipy import stats
import warnings


class CMBPlotter:
    """
    Comprehensive plotting utilities for CMB analysis results.

    This class provides methods for creating publication-quality plots of:
    - CMB power spectra
    - Parameter constraints
    - Residuals analysis
    - Theory comparisons
    """

    def __init__(self) -> None:
        """Initialize plotting configuration."""
        self.setup_style()
        self.colors = sns.color_palette("colorblind")
        self.fig_size = (12, 8)

    def setup_style(self) -> None:
        """Configure matplotlib style for publication-quality plots."""
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'text.usetex': True,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def plot_power_spectra(self, theory: Dict[str, ArrayLike],
                           data: Dict[str, ArrayLike],
                           errors: Dict[str, ArrayLike],
                           fig: Optional[Figure] = None) -> Figure:
        """
        Plot CMB power spectra with data and theory comparison.

        Parameters
        ----------
        theory : dict
            Dictionary containing theoretical spectra (TT, TE, EE)
        data : dict
            Dictionary containing observed spectra
        errors : dict
            Dictionary containing observational errors
        fig : Figure, optional
            Existing figure to plot on

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if fig is None:
            fig = plt.figure(figsize=(15, 10))

        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

        # TT spectrum
        ax_tt = fig.add_subplot(gs[0, 0])
        self._plot_spectrum(ax_tt, 'TT', theory, data, errors)

        # TE spectrum
        ax_te = fig.add_subplot(gs[0, 1])
        self._plot_spectrum(ax_te, 'TE', theory, data, errors)

        # EE spectrum
        ax_ee = fig.add_subplot(gs[1, :])
        self._plot_spectrum(ax_ee, 'EE', theory, data, errors)

        # Adjust layout
        plt.tight_layout()
        return fig

    def _plot_spectrum(self, ax: Axes, spec_type: str,
                       theory: Dict[str, ArrayLike],
                       data: Dict[str, ArrayLike],
                       errors: Dict[str, ArrayLike]) -> None:
        """
        Plot individual power spectrum.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object
        spec_type : str
            Type of spectrum (TT, TE, or EE)
        theory : dict
            Theoretical spectra
        data : dict
            Observed spectra
        errors : dict
            Observational errors
        """
        ell = np.arange(len(theory[f'cl_{spec_type.lower()}']))

        # Plot data points with errors
        ax.errorbar(ell, data[f'cl_{spec_type.lower()}'],
                    yerr=errors[f'cl_{spec_type.lower()}'],
                    fmt='k.', alpha=0.3, label='Data')

        # Plot theory curve
        ax.plot(ell, theory[f'cl_{spec_type.lower()}'],
                color=self.colors[0], label='Theory')

        # Customize plot
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(rf'$D_{{\ell}}^{{{spec_type}}}$ [$\mu$K$^2$]')

        if spec_type != 'TE':
            ax.set_yscale('log')
        ax.set_xscale('log')

        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_residuals(self, theory: Dict[str, ArrayLike],
                       data: Dict[str, ArrayLike],
                       errors: Dict[str, ArrayLike]) -> Figure:
        """
        Plot residuals between theory and data.

        Parameters
        ----------
        theory : dict
            Theoretical spectra
        data : dict
            Observed spectra
        errors : dict
            Observational errors

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        for i, spec_type in enumerate(['TT', 'TE', 'EE']):
            spec_key = f'cl_{spec_type.lower()}'
            residuals = (data[spec_key] - theory[spec_key]) / errors[spec_key]

            axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[i].fill_between(np.arange(len(residuals)),
                                 -1, 1, color='gray', alpha=0.2)
            axes[i].fill_between(np.arange(len(residuals)),
                                 -2, 2, color='gray', alpha=0.1)

            axes[i].plot(np.arange(len(residuals)), residuals,
                         'k.', alpha=0.5)

            axes[i].set_ylabel(f'{spec_type} Residuals (sigma)')
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel(r'$\ell$')
        plt.tight_layout()
        return fig

    def plot_corner(self, samples: ArrayLike,
                    labels: List[str],
                    truths: Optional[ArrayLike] = None) -> Figure:
        """
        Create corner plot for parameter constraints.

        Parameters
        ----------
        samples : array-like
            MCMC samples
        labels : list
            Parameter labels
        truths : array-like, optional
            True parameter values

        Returns
        -------
        Figure
            Corner plot figure
        """
        # Create corner plot with customization
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            truths=truths,
            truth_color='red',
            plot_datapoints=True,
            fill_contours=True,
            levels=(0.68, 0.95),
            color=self.colors[0],
            hist_kwargs={'color': self.colors[0]},
            smooth=1
        )

        return fig

    def plot_best_fit_comparison(self, ell: ArrayLike,
                                 data: ArrayLike,
                                 best_fit: ArrayLike,
                                 error: ArrayLike,
                                 title: str = "") -> Figure:
        """
        Plot comparison between data and best-fit model.

        Parameters
        ----------
        ell : array-like
            Multipole moments
        data : array-like
            Observed spectrum
        best_fit : array-like
            Best-fit theoretical spectrum
        error : array-like
            Observational errors
        title : str, optional
            Plot title

        Returns
        -------
        Figure
            Comparison plot figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                       height_ratios=[3, 1],
                                       sharex=True)

        # Upper panel: data and best fit
        ax1.errorbar(ell, data, yerr=error,
                     fmt='k.', alpha=0.3, label='Data')
        ax1.plot(ell, best_fit, color=self.colors[0],
                 label='Best fit')

        ax1.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title(title)

        # Lower panel: residuals
        residuals = (data - best_fit) / error
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(ell, -1, 1, color='gray', alpha=0.2)
        ax2.plot(ell, residuals, 'k.', alpha=0.5)

        ax2.set_xlabel(r'$\ell$')
        ax2.set_ylabel(r'Residuals ($\sigma$)')

        plt.tight_layout()
        return fig

    def plot_theory_variation(self, ell: ArrayLike,
                              samples: ArrayLike,
                              compute_theory: callable,
                              n_curves: int = 100) -> Figure:
        """
        Plot theory variations from MCMC samples.

        Parameters
        ----------
        ell : array-like
            Multipole moments
        samples : array-like
            MCMC samples
        compute_theory : callable
            Function to compute theory given parameters
        n_curves : int, optional
            Number of curves to plot

        Returns
        -------
        Figure
            Theory variation plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Randomly select samples
        indices = np.random.choice(len(samples), n_curves)

        # Plot theory curves
        for i in indices:
            params = samples[i]
            theory = compute_theory(params)
            ax.plot(ell, theory, color=self.colors[0],
                    alpha=0.1)

        # Customize plot
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        return fig

    def save_publication_plots(self, filename: str,
                               fig: Figure,
                               dpi: int = 300) -> None:
        """
        Save publication-quality plots.

        Parameters
        ----------
        filename : str
            Output filename
        fig : Figure
            Figure to save
        dpi : int, optional
            Resolution in dots per inch
        """
        # Save in multiple formats
        fig.savefig(f"{filename}.pdf", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches='tight')
