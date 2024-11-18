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
        self.colors = {
            'tt': '#1f77b4',  # blue
            'te': '#2ca02c',  # green
            'ee': '#ff7f0e'   # orange
        }
        self.ylims = {
            'tt': (1e2, 6e3),    # Updated TT limits
            'te': (-140, 140),   # Updated TE limits
            'ee': (1e-1, 1e2)    # Updated EE limits
        }

    def setup_style(self) -> None:
        """Configure matplotlib style for Planck-like plots."""
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
            'axes.grid': True,
            'grid.alpha': 0.3,
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
        """Plot CMB power spectra with Planck-like formatting."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Get ℓ range starting from ℓ=2
        ell = np.arange(2, len(theory['cl_tt']) + 2)
        ell_factor = ell * (ell + 1) / (2 * np.pi)

        # TT spectrum (top panel)
        ax = axes[0]
        theory_tt = theory['cl_tt'] * ell_factor
        data_tt = data['cl_tt'] * ell_factor[: len(data['cl_tt'])]
        errors_tt = errors['cl_tt'] * ell_factor[: len(errors['cl_tt'])]

        ax.set_yscale('log')
        ax.plot(ell, theory_tt, color=self.colors['tt'], label='Theory')
        ax.errorbar(ell[:len(data_tt)], data_tt, yerr=errors_tt,
                    fmt='.', color='gray', alpha=0.5, label='Data')
        ax.set_ylabel(r'$D_\ell^{TT}$ [$\mu$K$^2$]')
        ax.set_title('Temperature Power Spectrum (TT)')
        ax.set_ylim(self.ylims['tt'])
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
        ax.set_ylim(self.ylims['te'])

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
        ax.set_ylim(self.ylims['ee'])

        # Common x-axis settings
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlim(2, 2500)  # Standard Planck ℓ range
            ax.grid(True, alpha=0.3, which='both')

        axes[-1].set_xlabel('Multipole $\ell$')

        plt.tight_layout()
        return fig

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

    def plot_residuals(self, theory: Dict[str, np.ndarray],
                       data: Dict[str, np.ndarray],
                       errors: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot residuals with Planck-like formatting."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Get ℓ range starting from ℓ=2
        ell = np.arange(2, len(theory['cl_tt']) + 2)

        specs = ['tt', 'te', 'ee']
        titles = ['TT', 'TE', 'EE']

        for idx, (spec, title) in enumerate(zip(specs, titles)):
            ax = axes[idx]
            spec_key = f'cl_{spec}'

            # Get data length and truncate theory to match
            data_length = len(data[spec_key])
            theory_trunc = theory[spec_key][:data_length]

            # Calculate normalized residuals
            residuals = (data[spec_key] - theory_trunc) / errors[spec_key]

            # Calculate χ²/dof
            chi2 = np.sum(residuals**2)
            dof = len(residuals)
            chi2_dof = chi2 / dof

            # Plot residuals
            ax.plot(ell[:data_length], residuals,
                    '.', color=self.colors[spec], alpha=0.5)

            # Set scales and limits
            ax.set_xscale('log')
            ax.set_xlim(2, 2500)
            ax.set_ylim(-4, 4)

            # Labels and formatting
            ax.set_ylabel(fr'$\Delta D_{{\ell}}^{{{title}}}/\sigma$')
            ax.set_title(f'{title} Residuals (χ²/dof = {chi2_dof:.2f})')

            # Add zero line and grid
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, which='both')

        axes[-1].set_xlabel('Multipole $\ell$')
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
