"""
Implementation of CMB power spectrum calculations.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate
import warnings

from .transfer import CosmicTransferFunctions


class PowerSpectrumCalculator:
    """Calculator for CMB power spectra (TT, TE, EE)."""

    def __init__(self, transfer_functions: Optional[CosmicTransferFunctions] = None) -> None:
        """Initialize power spectrum calculator."""
        self.transfer = transfer_functions or CosmicTransferFunctions()
        self.ell_max = 2500
        self.k_max = 10.0
        self.setup_integration()

    def setup_integration(self) -> None:
        """Set up integration grids and weights."""
        # Set up k integration grid with adaptive sampling
        k_low = np.logspace(-4, -2, 1000)  # Dense sampling at low k
        k_mid = np.logspace(-2, 0, 2000)   # Medium sampling
        k_high = np.logspace(0, np.log10(self.k_max), 1000)  # Sparse at high k
        self.k_grid = np.unique(np.concatenate([k_low, k_mid, k_high]))
        self.ell = np.arange(2, self.ell_max + 1)

    def compute_all_spectra(self, params: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Compute spectra with improved numerical stability."""
        try:
            k = self.k_grid

            # Get transfer functions with stability checks
            T_m = np.nan_to_num(self.transfer.matter_transfer(k, params))
            T_r = np.nan_to_num(self.transfer.radiation_transfer(k, params))

            # Compute primordial spectrum with logarithmic handling
            ln_As = params['ln10As'] + np.log(1e-10)
            n_s = params['ns']
            ln_P_prim = ln_As + (n_s - 1) * np.log(k/0.05)
            P_prim = np.exp(ln_P_prim)

            # Add reionization effects
            tau = params['tau']
            damping = np.exp(-2*tau)

            # Compute spectra in log space where appropriate
            cl_tt = self._compute_tt_spectrum(k, P_prim, T_m, T_r, damping)
            cl_ee = self._compute_ee_spectrum(k, P_prim, T_m, T_r, damping, tau)
            cl_te = self._compute_te_spectrum(cl_tt, cl_ee)

            # Apply physical constraints
            cl_tt = np.maximum(cl_tt, 1e-30)
            cl_ee = np.maximum(cl_ee, 1e-30)
            cl_te = np.clip(cl_te, -np.sqrt(cl_tt * cl_ee), np.sqrt(cl_tt * cl_ee))

            return cl_tt, cl_ee, cl_te

        except Exception as e:
            print(f"Power spectrum computation error: {str(e)}")
            return [np.full(len(self.ell), np.nan) for _ in range(3)]

    def _compute_tt_spectrum(self, k: ArrayLike, P_prim: ArrayLike,
                             T_m: ArrayLike, T_r: ArrayLike, damping: float) -> ArrayLike:
        """Improved temperature spectrum computation."""
        try:
            # Use log-space integration where possible
            ln_integrand = np.log(P_prim) + 2*np.log(np.abs(T_m)
                                                     ) + 2*np.log(np.abs(T_r))
            integrand = np.exp(ln_integrand)

            # Vectorized integration over k
            cl = np.zeros(len(self.ell))
            for i, l in enumerate(self.ell):
                weight = self._spherical_bessel(l, k)
                cl[i] = damping * integrate.simpson(y=integrand * weight**2, x=k)

            return cl * 2 * np.pi * 1e10

        except Exception as e:
            print(f"Error in TT spectrum computation: {str(e)}")
            return np.full(len(self.ell), np.nan)

    def _compute_ee_spectrum(self, k: ArrayLike, P_prim: ArrayLike,
                             T_m: ArrayLike, T_r: ArrayLike, damping: float,
                             tau: float) -> ArrayLike:
        """Compute E-mode polarization power spectrum."""
        try:
            # Polarization source
            pol_source = (1 - np.exp(-tau))**2

            # Use log-space computation where possible
            ln_integrand = np.log(P_prim) + 2*np.log(np.abs(T_m)
                                                     ) + 2*np.log(np.abs(T_r))
            integrand = np.exp(ln_integrand) * pol_source

            # Vectorized integration over k
            cl = np.zeros(len(self.ell))
            for i, l in enumerate(self.ell):
                weight = self._spherical_bessel(l, k)
                cl[i] = damping * integrate.simpson(y=integrand * weight**2, x=k)

            return np.maximum(cl * 2 * np.pi * 1e10, 1e-30)

        except Exception as e:
            print(f"Error in EE spectrum computation: {str(e)}")
            return np.full(len(self.ell), np.nan)

    def _compute_te_spectrum(self, cl_tt: ArrayLike, cl_ee: ArrayLike) -> ArrayLike:
        """Compute temperature-E-mode cross spectrum."""
        try:
            # Ensure proper sign handling
            sign_tt = np.sign(cl_tt)
            amplitude = np.sqrt(np.abs(cl_tt * cl_ee))

            # Apply physical constraints
            te_spec = sign_tt * amplitude
            max_te = np.sqrt(np.abs(cl_tt * cl_ee))

            return np.clip(te_spec, -max_te, max_te)

        except Exception as e:
            print(f"Error in TE spectrum computation: {str(e)}")
            return np.full(len(self.ell), np.nan)

    def _spherical_bessel(self, l: int, k: ArrayLike) -> ArrayLike:
        """Compute spherical Bessel function with improved stability."""
        try:
            x = k * self.transfer.r_s
            y = np.zeros_like(x)

            # Small x approximation
            small_x = x < 0.1
            y[small_x] = x[small_x]**l / (2*l + 1)

            # Large x approximation
            large_x = x >= 0.1
            y[large_x] = np.sin(x[large_x] - l*np.pi/2) / (x[large_x] + 1e-30)

            return y

        except Exception as e:
            print(f"Error in Bessel function computation: {str(e)}")
            return np.zeros_like(k)

    def get_dimensionless_power(self, cl_values: ArrayLike) -> ArrayLike:
        """Convert C_l to dimensionless power spectrum D_l."""
        return self.ell * (self.ell + 1) * cl_values / (2 * np.pi)

    def compute_chi_square(self, theory: ArrayLike, data: ArrayLike,
                           error: ArrayLike) -> float:
        """Compute χ² between theory and data."""
        # Ensure all arrays have matching lengths
        min_len = min(len(theory), len(data), len(error))
        theory = theory[:min_len]
        data = data[:min_len]
        error = error[:min_len]

        # Mask out invalid points
        mask = (error > 0) & np.isfinite(data) & np.isfinite(error)
        if not np.any(mask):
            return np.inf

        return np.sum(((data[mask] - theory[mask]) / error[mask])**2)
