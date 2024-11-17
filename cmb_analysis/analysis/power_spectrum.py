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
    """
    Calculator for CMB power spectra (TT, TE, EE).

    This class handles the computation of theoretical CMB power spectra,
    including:
    - Temperature auto-correlation (TT)
    - Temperature-E-mode cross-correlation (TE)
    - E-mode auto-correlation (EE)
    """

    def __init__(self, transfer_functions: Optional[CosmicTransferFunctions] = None) -> None:
        """
        Initialize power spectrum calculator.

        Parameters
        ----------
        transfer_functions : CosmicTransferFunctions, optional
            Pre-initialized transfer function calculator
        """
        self.transfer = transfer_functions or CosmicTransferFunctions()
        self.ell_max = 2500
        self.k_max = 10.0  # Maximum k in h/Mpc
        self.setup_integration()

    def setup_integration(self) -> None:
        """Set up integration grids and weights."""
        # Set up k integration grid
        self.k_grid = np.logspace(-4, np.log10(self.k_max), 1000)

        # Set up ell values
        self.ell = np.arange(2, self.ell_max + 1)

    def compute_all_spectra(self, params: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Compute all CMB power spectra.

        Parameters
        ----------
        params : dict
            Cosmological parameters

        Returns
        -------
        tuple
            TT, TE, and EE power spectra
        """
        try:
            # Get transfer functions
            k = self.k_grid
            T_m = self.transfer.matter_transfer(k, params)
            T_r = self.transfer.radiation_transfer(k, params)

            # Add reionization effects
            tau = params['tau']
            damping = np.exp(-2*tau)

            # Primordial spectrum
            A_s = np.exp(params['ln10As']) * 1e-10
            n_s = params['ns']
            P_prim = A_s * (k/0.05)**(n_s-1)

            # Compute spectra with numerical stability
            cl_tt = self._compute_tt_spectrum(k, P_prim, T_m, T_r, damping)
            cl_ee = self._compute_ee_spectrum(k, P_prim, T_m, T_r, damping, tau)
            cl_te = self._compute_te_spectrum(cl_tt, cl_ee)

            return cl_tt, cl_ee, cl_te

        except Exception as e:
            warnings.warn(f"Error in power spectrum computation: {e}")
            return (np.full(len(self.ell), np.nan),
                    np.full(len(self.ell), np.nan),
                    np.full(len(self.ell), np.nan))

    def _compute_tt_spectrum(self, k: ArrayLike, P_prim: ArrayLike,
                             T_m: ArrayLike, T_r: ArrayLike, damping: float) -> ArrayLike:
        """
        Compute temperature power spectrum.

        Parameters
        ----------
        k : array-like
            Wavenumbers
        P_prim : array-like
            Primordial power spectrum
        T_m : array-like
            Matter transfer function
        T_r : array-like
            Radiation transfer function
        damping : float
            Reionization damping factor

        Returns
        -------
        array-like
            TT power spectrum
        """
        integrand = P_prim * T_m**2 * T_r**2

        # Perform k integration for each ell
        cl = np.zeros(len(self.ell))
        for i, l in enumerate(self.ell):
            weight = self._spherical_bessel(l, k)
            cl[i] = damping * integrate.simps(integrand * weight**2, k)

        return np.maximum(cl * 1e12, 1e-30)  # Convert to μK²

    def _compute_ee_spectrum(self, k: ArrayLike, P_prim: ArrayLike,
                             T_m: ArrayLike, T_r: ArrayLike, damping: float,
                             tau: float) -> ArrayLike:
        """
        Compute E-mode polarization power spectrum.

        Parameters
        ----------
        k : array-like
            Wavenumbers
        P_prim : array-like
            Primordial power spectrum
        T_m : array-like
            Matter transfer function
        T_r : array-like
            Radiation transfer function
        damping : float
            Reionization damping factor
        tau : float
            Optical depth to reionization

        Returns
        -------
        array-like
            EE power spectrum
        """
        # Polarization source
        pol_source = (1 - np.exp(-tau))**2
        integrand = P_prim * T_m**2 * T_r**2 * pol_source

        # Perform k integration for each ell
        cl = np.zeros(len(self.ell))
        for i, l in enumerate(self.ell):
            weight = self._spherical_bessel(l, k)
            cl[i] = damping * integrate.simps(integrand * weight**2, k)

        return np.maximum(cl * 1e12, 1e-30)  # Convert to μK²

    def _compute_te_spectrum(self, cl_tt: ArrayLike, cl_ee: ArrayLike) -> ArrayLike:
        """
        Compute temperature-E-mode cross spectrum.

        Parameters
        ----------
        cl_tt : array-like
            Temperature power spectrum
        cl_ee : array-like
            E-mode power spectrum

        Returns
        -------
        array-like
            TE cross-spectrum
        """
        return np.sign(cl_tt) * np.sqrt(np.abs(cl_tt * cl_ee))

    def _spherical_bessel(self, l: int, k: ArrayLike) -> ArrayLike:
        """
        Compute spherical Bessel function for power spectrum integration.

        Parameters
        ----------
        l : int
            Multipole moment
        k : array-like
            Wavenumbers

        Returns
        -------
        array-like
            Spherical Bessel function values
        """
        x = k * self.transfer.r_s

        # Asymptotic forms for numerical stability
        y = np.zeros_like(x)

        # Small x approximation
        small_x = x < 0.1
        y[small_x] = x[small_x]**l / (2*l + 1)

        # Large x approximation
        large_x = x >= 0.1
        y[large_x] = np.sin(x[large_x] - l*np.pi/2) / x[large_x]

        return y

    def get_dimensionless_power(self, cl_values: ArrayLike) -> ArrayLike:
        """
        Convert C_l to dimensionless power spectrum D_l.

        Parameters
        ----------
        cl_values : array-like
            C_l power spectrum

        Returns
        -------
        array-like
            D_l = l(l+1)C_l/(2π)
        """
        return self.ell * (self.ell + 1) * cl_values / (2 * np.pi)

    def compute_chi_square(self, theory: ArrayLike, data: ArrayLike,
                           error: ArrayLike) -> float:
        """
        Compute χ² between theory and data.

        Parameters
        ----------
        theory : array-like
            Theoretical power spectrum
        data : array-like
            Observed power spectrum
        error : array-like
            Uncertainties in observed spectrum

        Returns
        -------
        float
            χ² value
        """
        return np.sum(((data - theory) / error)**2)
