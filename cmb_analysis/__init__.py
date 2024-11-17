"""
Physical constants and conversion factors for CMB analysis.

This module provides a comprehensive set of physical constants,
unit conversions, and cosmological parameters commonly used in
CMB analysis. All values are in SI units unless otherwise noted.
"""

import numpy as np

# Physical constants (SI units)
SPEED_OF_LIGHT = 2.99792458e8  # Speed of light (m/s)
GRAVITATIONAL_CONSTANT = 6.67430e-11  # Newton's gravitational constant (m³/kg/s²)
PLANCK_CONSTANT = 6.62607015e-34  # Planck constant (J⋅s)
BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant (J/K)
ELECTRON_MASS = 9.1093837015e-31  # Electron mass (kg)
PROTON_MASS = 1.67262192369e-27  # Proton mass (kg)
THOMSON_CROSS_SECTION = 6.6524587321e-29  # Thomson cross-section (m²)

# Astronomical constants
ASTRONOMICAL_UNIT = 1.495978707e11  # Astronomical Unit (m)
PARSEC = 3.085677581e16  # Parsec (m)
MEGAPARSEC = 3.085677581e22  # Megaparsec (m)
SOLAR_MASS = 1.98847e30  # Solar mass (kg)
YEAR_IN_SECONDS = 365.25 * 24 * 3600  # Julian year (s)

# CMB specific constants
CMB_TEMPERATURE = 2.7255  # CMB temperature (K)
DELTA_NEFF = 3.046  # Effective number of neutrino species
YHE = 0.2454  # Primordial helium abundance

# Conversion factors
KM_PER_MPC = 3.085677581e19  # km per Mpc
KM_PER_S_TO_MPC_PER_GYR = 1.022712165045695  # Convert km/s to Mpc/Gyr
EV_TO_KELVIN = 11604.525  # Convert eV to Kelvin
KELVIN_TO_EV = 8.617333262e-5  # Convert Kelvin to eV

# Planck 2018 cosmological parameters (fiducial values)
PLANCK_H0 = 67.32  # Hubble constant (km/s/Mpc)
PLANCK_OMEGA_B = 0.02237  # Physical baryon density
PLANCK_OMEGA_CDM = 0.1200  # Physical cold dark matter density
PLANCK_OMEGA_M = 0.3158  # Total matter density parameter
PLANCK_OMEGA_LAMBDA = 0.6842  # Dark energy density parameter
PLANCK_TAU = 0.0544  # Optical depth to reionization
PLANCK_NS = 0.9649  # Scalar spectral index
PLANCK_LN10AS = 3.044  # Log of primordial amplitude
PLANCK_SIGMA8 = 0.8111  # Power spectrum normalization

# Recombination and reionization parameters
Z_REIO = 7.67  # Reionization redshift
Z_STAR = 1089.80  # Last scattering surface redshift
Z_DRAG = 1059.94  # Drag epoch redshift
SOUND_HORIZON_STAR = 144.43  # Sound horizon at last scattering (Mpc)
SOUND_HORIZON_DRAG = 147.09  # Sound horizon at drag epoch (Mpc)
ANGULAR_DIAMETER_DISTANCE_STAR = 13.8688e3  # Angular diameter distance to LSS (Mpc)

# Numerical constants
LMAX_DEFAULT = 2500  # Default maximum multipole
KMAX_DEFAULT = 10.0  # Default maximum wavenumber (h/Mpc)
ZMAX_DEFAULT = 1100  # Default maximum redshift
INTEGRATION_EPSREL = 1e-8  # Default relative error for integration

# Unit conversions for power spectra
MU_K_TO_K = 1e-6  # Convert μK to K
K_TO_MU_K = 1e6  # Convert K to μK
MU_K_SQ = K_TO_MU_K**2  # Convert K² to μK²


class Constants:
    """
    Class containing all physical constants and conversion factors.
    This provides a namespace for constants and allows for type hints.
    """

    def __init__(self):
        """Initialize constants."""
        # Physical constants
        self.c = SPEED_OF_LIGHT
        self.G = GRAVITATIONAL_CONSTANT
        self.h = PLANCK_CONSTANT
        self.k_B = BOLTZMANN_CONSTANT
        self.m_e = ELECTRON_MASS
        self.m_p = PROTON_MASS
        self.sigma_T = THOMSON_CROSS_SECTION

        # Astronomical constants
        self.au = ASTRONOMICAL_UNIT
        self.pc = PARSEC
        self.Mpc = MEGAPARSEC
        self.M_sun = SOLAR_MASS
        self.year_seconds = YEAR_IN_SECONDS

        # CMB constants
        self.T_cmb = CMB_TEMPERATURE
        self.N_eff = DELTA_NEFF
        self.Y_He = YHE

        # Planck parameters
        self.H0 = PLANCK_H0
        self.omega_b = PLANCK_OMEGA_B
        self.omega_cdm = PLANCK_OMEGA_CDM
        self.Omega_m = PLANCK_OMEGA_M
        self.Omega_Lambda = PLANCK_OMEGA_LAMBDA
        self.tau = PLANCK_TAU
        self.n_s = PLANCK_NS
        self.ln10As = PLANCK_LN10AS
        self.sigma8 = PLANCK_SIGMA8

        # Derived quantities
        self.h = self.H0 / 100  # Dimensionless Hubble parameter
        self.age_universe = 13.797  # Age of universe in Gyr
        self.rho_crit = self._compute_critical_density()

    def _compute_critical_density(self) -> float:
        """
        Compute critical density of the universe.

        Returns
        -------
        float
            Critical density in kg/m³
        """
        H0_SI = self.H0 * 1000 / self.Mpc  # Convert to SI
        return 3 * H0_SI**2 / (8 * np.pi * self.G)

    def __str__(self) -> str:
        """String representation of constants."""
        return "CMB Analysis Physical Constants"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Constants(H0={self.H0:.2f}, T_cmb={self.T_cmb:.4f})"


# Create a single instance for import
constants = Constants()

# Make all constants available at module level
__all__ = [
    'constants',
    'SPEED_OF_LIGHT',
    'GRAVITATIONAL_CONSTANT',
    'PLANCK_CONSTANT',
    'BOLTZMANN_CONSTANT',
    'CMB_TEMPERATURE',
    'THOMSON_CROSS_SECTION',
    'PLANCK_H0',
    'PLANCK_OMEGA_B',
    'PLANCK_OMEGA_CDM',
    'PLANCK_TAU',
    'PLANCK_NS',
    'PLANCK_LN10AS'
]
