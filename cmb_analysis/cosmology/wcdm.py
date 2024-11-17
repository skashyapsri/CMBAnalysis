"""
Implementation of the wCDM cosmological model with constant dark energy equation of state.
"""

from typing import Dict, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from .base import CosmologyModel


class wCDM(CosmologyModel):
    """
    wCDM cosmological model with constant dark energy equation of state.

    This class extends the standard ΛCDM model by allowing the dark energy
    equation of state parameter w to vary (w = -1 corresponds to ΛCDM).

    Attributes
    ----------
    param_info : dict
        Dictionary containing parameter names, fiducial values, and uncertainties
        based on Planck 2018 results plus w
    """

    def __init__(self) -> None:
        """Initialize wCDM model with Planck 2018 parameters plus w."""
        super().__init__()

        # Planck 2018 parameters (mean, std) plus w
        self.param_info = {
            'H0':      (67.32, 0.54),
            'omega_b': (0.02237, 0.00015),
            'omega_cdm': (0.1200, 0.0012),
            'tau':     (0.0544, 0.0073),
            'ns':      (0.9649, 0.0042),
            'ln10As':  (3.044, 0.014),
            'w':       (-1.0, 0.1)      # Dark energy equation of state
        }

    def H(self, z: Union[float, ArrayLike], params: Dict[str, float]) -> Union[float, ArrayLike]:
        """
        Compute the Hubble parameter H(z) for wCDM.

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
        Uses the Friedmann equation for a flat wCDM universe:
        H(z) = H0 * sqrt(Ωm(1+z)³ + ΩDE(1+z)^(3(1+w)))
        """
        H0 = params['H0']
        omega_m = (params['omega_b'] + params['omega_cdm']) / (H0/100)**2
        w = params['w']

        return H0 * np.sqrt(
            omega_m * (1+z)**3 +
            (1-omega_m) * (1+z)**(3*(1+w))
        )

    def angular_diameter_distance(self, z: float, params: Dict[str, float]) -> float:
        """
        Compute the angular diameter distance for wCDM.

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
        Computes D_A = (c/H0) * χ(z
        """
        # Convert H0 to s^-1
        H0 = params['H0'] * 1000 / self.Mpc

        def integrand(z):
            return 1 / self.H(z, params)

        # Compute comoving distance
        chi, _ = integrate.quad(integrand, 0, z)
        chi *= self.c / H0

        # Convert to angular diameter distance
        D_A = chi / (1 + z)

        return D_A / self.Mpc
