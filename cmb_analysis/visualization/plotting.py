import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import seaborn as sns
from typing import Dict, Optional
from numpy.typing import ArrayLike


class CMBPlotter:
    """Enhanced CMB power spectra plotter with improved formatting."""

    def __init__(self) -> None:
        """Initialize plotting configuration."""
        self.setup_style()
        self.fig_size = (12, 15)
        self.colors = {
            'tt': '#1f77b4',  # blue
            'te': '#2ca02c',  # green
            'ee': '#ff7f0e'   # orange
        }
        # Add spectrum-specific limits
        self.ylims = {
            'tt': (10, 1e4),
            'te': (-150, 150),
            'ee': (1e-1, 1e2)
        }

    def setup_style(self) -> None:
        """Configure matplotlib style for publication quality plots."""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (12, 15),
            'figure.dpi': 100,
            'text.usetex': False,
            'mathtext.default': 'regular',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2
        })

    def _validate_data(self, theory: Dict[str, np.ndarray],
                       data: Dict[str, np.ndarray],
                       errors: Dict[str, np.ndarray]) -> None:
        """Validate input data consistency."""
        for spec in ['tt', 'te', 'ee']:
            key = f'cl_{spec}'
            if key not in theory or key not in data or key not in errors:
                raise ValueError(f"Missing {key} in input data")
            if len(data[key]) != len(errors[key]):
                raise ValueError(f"Data and error length mismatch for {key}")
            if len(theory[key]) < len(data[key]):
                raise ValueError(f"Theory spectrum too short for {key}")

    def plot_theory_spectrum(self, ax, spec_type: str, theory: Dict[str, ArrayLike], color: str = 'red') -> None:
        spec_key = f'cl_{spec_type.lower()}'

        # Separate ell ranges for theory
        ell_theory = np.arange(len(theory[spec_key]))

        # Calculate D_l factors
        dl_factor_theory = ell_theory * (ell_theory + 1) / (2 * np.pi)

        # Apply conversions
        theory_dl = theory[spec_key] * dl_factor_theory

        # Plot theory curve
        ax.plot(ell_theory, theory_dl, color=color, label=r'$\Lambda$CDM fit')

    def plot_power_spectra(self, theory: Dict[str, np.ndarray],
                           data: Dict[str, np.ndarray],
                           errors: Dict[str, np.ndarray]) -> plt.Figure:
        """
            Plot CMB power spectra with proper scaling and formatting.

            Parameters
            ----------
            theory : dict
                Dictionary containing theoretical Cℓ values
            data : dict
                Dictionary containing observed Cℓ values
            errors : dict
                Dictionary containing error estimates
            """
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Common x-axis settings
        for ax in axes:
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Multipole ℓ')

        # Get ℓ range
        ell = np.arange(len(theory['cl_tt']))
        ell_factor = ell * (ell + 1) / (2 * np.pi)

        # TT spectrum (top panel)
        ax = axes[0]
        # Convert to Dℓ = ℓ(ℓ+1)Cℓ/(2π)
        theory_tt = theory['cl_tt'] * ell_factor
        data_tt = data['cl_tt'] * ell_factor[: len(data['cl_tt'])]
        errors_tt = errors['cl_tt'] * ell_factor[: len(errors['cl_tt'])]

        ax.set_yscale('log')
        ax.plot(ell, theory_tt, color=self.colors['tt'], label='Theory')
        ax.errorbar(ell[:len(data_tt)], data_tt, yerr=errors_tt,
                    fmt='.', color='gray', alpha=0.5, label='Data')
        ax.set_ylabel(r'$D_\ell^{TT}$ [$\mu$K$^2$]')
        ax.set_title('Temperature Power Spectrum (TT)')
        ax.legend()

        # TE spectrum (middle panel)
        ax = axes[1]
        theory_te = theory['cl_te'] * ell_factor
        data_te = data['cl_te'] * ell_factor[: len(data['cl_te'])]
        errors_te = errors['cl_te'] * ell_factor[: len(errors['cl_te'])]

        ax.plot(ell, theory_te, color=self.colors['te'])
        ax.errorbar(ell[:len(data_te)], data_te, yerr=errors_te,
                    fmt='.', color='gray', alpha=0.5)
        ax.set_ylabel(r'$D_\ell^{TE}$ [$\mu$K$^2$]')
        ax.set_title('Temperature-E-mode Cross Spectrum (TE)')

        # EE spectrum (bottom panel)
        ax = axes[2]
        theory_ee = theory['cl_ee'] * ell_factor
        data_ee = data['cl_ee'] * ell_factor[: len(data['cl_ee'])]
        errors_ee = errors['cl_ee'] * ell_factor[: len(errors['cl_ee'])]

        ax.set_yscale('log')
        ax.plot(ell, theory_ee, color=self.colors['ee'])
        ax.errorbar(ell[:len(data_ee)], data_ee, yerr=errors_ee,
                    fmt='.', color='gray', alpha=0.5)
        ax.set_ylabel(r'$D_\ell^{EE}$ [$\mu$K$^2$]')
        ax.set_title('E-mode Power Spectrum (EE)')

        # Adjust layout
        plt.tight_layout()
        return fig

    def _plot_spectrum(self, ax, spec_type: str,
                       theory: Dict[str, ArrayLike],
                       data: Dict[str, ArrayLike],
                       errors: Dict[str, ArrayLike]) -> None:
        """
        Plot individual power spectrum with enhanced formatting.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on
        spec_type : str
            Type of spectrum (TT, TE, or EE)
        theory : dict
            Theoretical power spectrum
        data : dict
            Observed power spectrum
        errors : dict
            Measurement uncertainties
        """
        spec_key = f'cl_{spec_type.lower()}'

        # Separate ell ranges for theory and data
        ell_theory = np.arange(len(theory[spec_key]))
        ell_data = np.arange(len(data[spec_key]))

        # Calculate D_l factors separately
        dl_factor_theory = ell_theory * (ell_theory + 1) / (2 * np.pi)
        dl_factor_data = ell_data * (ell_data + 1) / (2 * np.pi)

        # Apply conversions separately
        theory_dl = theory[spec_key] * dl_factor_theory
        data_dl = data[spec_key] * dl_factor_data
        errors_dl = errors[spec_key] * dl_factor_data  # Using same ell as data

        # Plot theory curve
        ax.plot(ell_theory, theory_dl,
                color=self.colors[0],
                label='Theory',
                linewidth=2,
                zorder=2)

        # Plot data points with error bars
        ax.errorbar(ell_data[::20], data_dl[::20],  # Plot fewer points for clarity
                    yerr=errors_dl[::20],
                    fmt='k.',
                    markersize=4,
                    alpha=0.5,
                    label='Data',
                    zorder=1)

        # Set scales
        ax.set_xscale('log')
        if spec_type != 'TE':
            ax.set_yscale('log')

        # Set appropriate axis limits and labels
        ax.set_xlim(2, 2500)
        if spec_type == 'TT':
            ax.set_ylim(10, 10000)
        elif spec_type == 'EE':
            ax.set_ylim(0.1, 100)
        elif spec_type == 'TE':
            ax.set_ylim(-200, 200)

        ax.set_xlabel('Multipole ℓ')
        ax.set_ylabel(f'D_ℓ {spec_type} [μK²]')

        # Enhance grid and legend
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(frameon=True, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98))

        # Add minor ticks
        ax.minorticks_on()

        # Enhance spine visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    def plot_residuals(self, theory: Dict[str, np.ndarray],
                       data: Dict[str, np.ndarray],
                       errors: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot residuals with proper χ² calculation."""
        # Validate input data
        self._validate_data(theory, data, errors)

        fig, axes = plt.subplots(3, 1, figsize=self.fig_size, sharex=True)

        specs = ['tt', 'te', 'ee']
        for idx, spec in enumerate(specs):
            ax = axes[idx]
            spec_key = f'cl_{spec}'

            # Get data length and ell range
            data_length = len(data[spec_key])
            ell = np.arange(2, data_length + 2)  # Start from ℓ=2

            # Calculate ell factor
            ell_factor = ell * (ell + 1) / (2 * np.pi)

            # Convert to Dℓ and compute residuals
            theory_dl = theory[spec_key][:data_length] * ell_factor
            data_dl = data[spec_key] * ell_factor
            errors_dl = errors[spec_key] * ell_factor

            # Compute normalized residuals
            residuals = (data_dl - theory_dl) / errors_dl

            # Calculate χ²/dof excluding NaN values
            valid_mask = np.isfinite(residuals)
            if np.any(valid_mask):
                chi2 = np.sum(residuals[valid_mask]**2)
                dof = np.sum(valid_mask)
                chi2_dof = chi2 / dof
            else:
                chi2_dof = np.nan

            # Plot residuals
            ax.errorbar(ell, residuals, yerr=1.0, fmt='.',
                        color=self.colors[spec], alpha=0.5)

            # Set scales and limits
            ax.set_xscale('log')
            ax.set_xlim(2, 2500)
            ax.set_ylim(-5, 5)

            # Labels and formatting
            ax.set_xlabel('Multipole ℓ')
            ax.set_ylabel(rf'$\Delta D_{{\ell}}^{{{spec.upper()}}}/\sigma$')
            ax.set_title(f'{spec.upper()} Residuals (χ²/dof = {chi2_dof:.2f})')

            # Add zero line and grid
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, which='both')
            ax.minorticks_on()

        plt.tight_layout()
        return fig


# Example usage:
if __name__ == "__main__":
    # Create mock data
    ell = np.arange(2, 2501)
    mock_theory = {
        'cl_tt': 1000 * (ell/100)**(-0.6) * np.exp(-(ell/1000)**2),
        'cl_ee': 100 * (ell/100)**(-0.6) * np.exp(-(ell/1000)**2),
        'cl_te': 300 * (ell/100)**(-0.6) * np.exp(-(ell/1000)**2)
    }

    mock_data = {key: val * (1 + 0.05 * np.random.randn(len(ell)))
                 for key, val in mock_theory.items()}
    mock_errors = {key: np.abs(0.1 * val)
                   for key, val in mock_theory.items()}

    # Create plot
    plotter = CMBPlotter()
    fig = plotter.plot_power_spectra(mock_theory, mock_data, mock_errors)
    plt.show()
