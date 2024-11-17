"""
Implementation of the standard ΛCDM cosmological model.
"""

from typing import Dict, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from .base import CosmologyModel


class LCDM(CosmologyModel):
    """
    Standard ΛCDM (Lambda Cold Dark Matter) cosmological model.

    This class implements the standard ΛCDM model with six parameters:
    - H0: Hubble constant
    - omega_b: Physical baryon density
    - omega_cdm: Physical cold dark matter density
    - tau: Optical depth to reionization
    - ns: Scalar spectral index
    - ln10As: Log of primordial amplitude

    The model assumes a flat universe with a cosmological constant.

    Attributes
    ----------
    param_info : dict
        Dictionary containing parameter names, fiducial values, and uncertainties
        based on Planck 2018 results
    """

    def __init__(self) -> None:
        """Initialize ΛCDM model with Planck 2018 parameters."""
        super().__init__()

        # Planck 2018 parameters (mean, std)
        self.param_info = {
            'H0':      (67.32, 0.54),
            'omega_b': (0.02237, 0.00015),
            'omega_cdm': (0.1200, 0.0012),
            'tau':     (0.0544, 0.0073),
            'ns':      (0.9649, 0.0042),
            'ln10As':  (3.044, 0.014)
        }

    def H(self, z: Union[float, ArrayLike], params: Dict[str, float]) -> Union[float, ArrayLike]:
        """
        Compute the Hubble parameter H(z) for ΛCDM.

        Parameters
        ----------
        z : float or array-like
            Redshift
        params : dict
            Cosmological parameters

        Returns
        -------
        float or array-like
            Hubble parameter in km/s/Mpc

        Notes
        -----
        Uses the Friedmann equation for a flat ΛCDM universe:
        H(z) = H0 * sqrt(Ωm(1+z)³ + ΩΛ)
        """
        H0 = params['H0']
        omega_m = (params['omega_b'] + params['omega_cdm']) / (H0/100)**2

        return H0 * np.sqrt(omega_m * (1+z)**3 + (1-omega_m))

    def angular_diameter_distance(self, z: float, params: Dict[str, float]) -> float:
        """
        Compute the angular diameter distance for ΛCDM.

        Parameters
        ----------
        z : float
            Redshift
        params : dict
            Cosmological parameters

        Returns
        -------
        float
            Angular diameter distance in Mpc

        Notes
        -----
        Computes D_A = (c/H0) * χ(z)/(1+z), where χ(z) is the comoving distance
        """
        if not self.validate_parameters(params):
            return np.nan

        # Convert H0 to s^-1
        H0 = params['H0'] * 1000 / self.Mpc

        try:
            # Compute comoving distance
            chi, _ = integrate.quad(
                lambda x: 1/self.H(x, params),
                0, z,
                epsrel=1e-8
            )
            chi *= self.c / H0  # Convert to meters

            # Convert to angular diameter distance
            D_A = chi / (1 + z)

            return D_A / self.Mpc  # Convert to Mpc

        except Exception as e:
            print(f"Error in angular diameter distance calculation: {e}")
            return np.nan

    def omega_m(self, params: Dict[str, float]) -> float:
        """
        Compute the total matter density parameter.

        Parameters
        ----------
        params : dict
            Cosmological parameters

        Returns
        -------
        float
            Total matter density parameter Ωm
        """
        H0 = params['H0']
        return (params['omega_b'] + params['omega_cdm']) / (H0/100)**2

    def age_of_universe(self, params: Dict[str, float]) -> float:
        """
        Compute the age of the universe in Gyr.

        Parameters
        ----------
        params : dict
            Cosmological parameters

        Returns
        -------
        float
            Age of the universe in Gyr
        """
        # Convert H0 to s^-1
        H0 = params['H0'] * 1000 / self.Mpc

        try:
            # Integrate 1/[(1+z)H(z)] from z=∞ (practically z=1000) to 0
            age, _ = integrate.quad(
                lambda z: 1/((1+z) * self.H(z, params)),
                0, 1000,
                epsrel=1e-8
            )

            # Convert to Gyr
            age *= self.c / H0 / (365.25 * 24 * 3600 * 1e9)

            return age

        except Exception as e:
            print(f"Error in age calculation: {e}")
            return np.nan
