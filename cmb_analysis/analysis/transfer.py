"""
Implementation of cosmic transfer functions for CMB analysis.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, interpolate
import warnings


class CosmicTransferFunctions:
    """
    Handles computation of cosmic transfer functions including effects of
    reionization, neutrinos, and other physical processes.

    This class implements various transfer functions needed for accurate
    CMB power spectrum calculations, including:
    - Matter transfer function
    - Radiation transfer function
    - Neutrino transfer function
    - Reionization effects
    """

    def __init__(self) -> None:
        """Initialize transfer function calculator with default parameters."""
        # Physical constants
        self.c = 2.99792458e8      # Speed of light (m/s)
        self.G = 6.67430e-11       # Gravitational constant
        self.h = 6.62607015e-34    # Planck constant
        self.k_B = 1.380649e-23    # Boltzmann constant
        self.T_cmb = 2.7255        # CMB temperature (K)

        # Recombination parameters
        self.z_rec = 1089.8
        self.tau_rec = 0.0
        self.r_s = 147.18  # Sound horizon at recombination (Mpc)
        self.theta_s = 0.0104  # Angular sound horizon

        # Reionization parameters
        self.z_reio_mean = 7.7
        self.delta_z_reio = 0.5

        # Neutrino parameters
        self.N_eff = 3.046
        self.m_nu_sum = 0.06  # eV

        self.setup_transfer_functions()

    def setup_transfer_functions(self) -> None:
        """Initialize transfer function components and lookup tables."""
        self.k_grid = np.logspace(-4, 2, 1000)  # k values in h/Mpc
        self._setup_matter_transfer()
        self._setup_radiation_transfer()

    def _setup_matter_transfer(self) -> None:
        """Initialize matter transfer function components."""
        # Set up interpolation tables for efficiency
        self.k_eq = 0.073  # Equality scale in h/Mpc
        self.alpha_nu = 0.0  # Neutrino suppression

    def _setup_radiation_transfer(self) -> None:
        """Initialize radiation transfer function components."""
        # Set up acoustic oscillation envelope
        self.silk_scale = 0.095  # Silk damping scale in h/Mpc

    def matter_transfer(self, k: ArrayLike, params: Dict[str, float]) -> ArrayLike:
        """
        Compute matter transfer function.

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc
        params : dict
            Cosmological parameters

        Returns
        -------
        array-like
            Matter transfer function T(k)
        """
        k = np.asarray(k)

        # Compute CDM + baryon transfer
        T_cb = self._compute_cdm_baryon_transfer(k, params)

        # Add neutrino effects if significant
        if self.m_nu_sum > 0.01:  # Only if neutrino mass is significant
            f_nu = self.m_nu_sum / (93.14 * params['omega_cdm'])
            T_nu = self._compute_neutrino_transfer(k, params)
            return (1 - f_nu) * T_cb + f_nu * T_nu

        return T_cb

    def _compute_cdm_baryon_transfer(self, k: ArrayLike,
                                     params: Dict[str, float]) -> ArrayLike:
        """
        Compute CDM + baryon transfer function.

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc
        params : dict
            Cosmological parameters

        Returns
        -------
        array-like
            CDM + baryon transfer function
        """
        # Simplified Eisenstein & Hu transfer function
        q = k * (params['H0']/100) / (params['omega_b']/0.02237)

        # Add BAO features
        alpha_c = (46.9 * params['omega_cdm'])**0.670 * \
            (1 + (32.1 * params['omega_cdm'])**(-0.532))
        beta_c = (12.0 * params['omega_cdm'])**0.424 * \
            (1 + (45.0 * params['omega_cdm'])**(-0.582))
        alpha_b = 2.07 * (params['omega_b']/0.02237)**0.294
        beta_b = 0.5 + params['omega_b']/0.02237 + (
            3 - 2 * params['omega_b']/0.02237) * np.sqrt(1 + (17.2 * params['omega_b']/0.02237)**2)

        f = 1 / (1 + (k * self.silk_scale)**4)  # Silk damping

        T = np.zeros_like(k)
        small_k = k < 0.01
        large_k = ~small_k

        # Small k limit
        T[small_k] = 1.0

        # Large k behavior with BAO
        T[large_k] = (np.log(2*np.e + 1.8*beta_c*q[large_k]) /
                      (np.log(2*np.e + 1.8*beta_c*q[large_k]) +
                      14.2*alpha_c*q[large_k]**2))

        # Add baryon acoustic features
        T *= (1 + (beta_b/(k*self.r_s))**3)**(-1)

        return T * f

    def _compute_neutrino_transfer(self, k: ArrayLike,
                                   params: Dict[str, float]) -> ArrayLike:
        """
        Compute neutrino transfer function.

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc
        params : dict
            Cosmological parameters

        Returns
        -------
        array-like
            Neutrino transfer function
        """
        # Compute phase space integral
        q = np.linspace(0, 10, 100)
        E = np.sqrt(q**2 + (self.m_nu_sum/(self.k_B*self.T_cmb))**2)
        f = 1 / (np.exp(q) + 1)  # Fermi-Dirac distribution

        # Compute suppression scale
        k_fs = 0.0178 * np.sqrt(params['omega_cdm']) * (self.m_nu_sum /
                                                        0.06) * ((1+self.z_rec)/3000) * params['H0']/100

        return np.exp(-(k/k_fs)**2)

    def radiation_transfer(self, k: ArrayLike, params: Dict[str, float]) -> ArrayLike:
        """
        Compute radiation transfer function.

        Parameters
        ----------
        k : array-like
            Wavenumbers in h/Mpc
        params : dict
            Cosmological parameters

        Returns
        -------
        array-like
            Radiation transfer function
        """
        k = np.asarray(k)

        # Acoustic oscillations
        krs = k * self.r_s
        T = np.cos(krs)

        # Silk damping
        k_d = 1 / self.silk_scale
        T *= np.exp(-(k/k_d)**2)

        return T

    def reionization_history(self, z: ArrayLike,
                             params: Dict[str, float]) -> ArrayLike:
        """
        Compute ionization fraction evolution.

        Parameters
        ----------
        z : array-like
            Redshifts
        params : dict
            Cosmological parameters

        Returns
        -------
        array-like
            Ionization fraction x_e(z)
        """
        z = np.asarray(z)
        tau = params['tau']

        # Tanh reionization model
        x_e = 0.5 * (1 + np.tanh((self.z_reio_mean - z)/self.delta_z_reio))

        # Normalize to get correct optical depth
        tau_norm = self.compute_optical_depth(z, x_e)
        x_e *= tau / tau_norm

        return x_e

    def compute_optical_depth(self, z: ArrayLike, x_e: ArrayLike) -> float:
        """
        Compute optical depth to reionization.

        Parameters
        ----------
        z : array-like
            Redshifts
        x_e : array-like
            Ionization fraction

        Returns
        -------
        float
            Optical depth τ
        """
        sigma_T = 6.6524587321e-29  # Thomson cross section
        n_e0 = 2.1927e-7  # Current electron density

        # Compute integral
        def integrand(z): return x_e * n_e0 * (1+z)**2 / np.sqrt(1+z)
        tau = sigma_T * self.c * integrate.simps(integrand(z), z)

        return tau

    def get_transfer_data(self) -> Dict[str, ArrayLike]:
        """
        Get precomputed transfer function data.

        Returns
        -------
        dict
            Dictionary containing k values and transfer functions
        """
        return {
            'k_grid': self.k_grid,
            'k_eq': self.k_eq,
            'silk_scale': self.silk_scale,
            'sound_horizon': self.r_s
        }
