"""
Base class for cosmological models in CMB analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
import warnings


class CosmologyModel(ABC):
    """
    Abstract base class for cosmological models.

    This class defines the interface for cosmological models used in CMB analysis.
    All specific model implementations (ΛCDM, wCDM, etc.) should inherit from this class.

    Attributes
    ----------
    c : float
        Speed of light in m/s
    G : float
        Gravitational constant in m³/kg/s²
    h : float
        Planck constant in J⋅s
    k_B : float
        Boltzmann constant in J/K
    T_cmb : float
        CMB temperature in K
    Mpc : float
        Megaparsec in meters
    rho_crit : float
        Critical density of the universe in kg/m³
    """

    def __init__(self) -> None:
        """Initialize physical constants and derived quantities."""
        # Physical constants (SI units)
        self.c = 2.99792458e8       # Speed of light (m/s)
        self.G = 6.67430e-11        # Gravitational constant
        self.h = 6.62607015e-34     # Planck constant
        self.k_B = 1.380649e-23     # Boltzmann constant
        self.T_cmb = 2.7255         # CMB temperature (K)
        self.Mpc = 3.085677581e22   # Megaparsec in meters

        # Derived quantities
        self.rho_crit = self._compute_critical_density()

        # Parameter information to be defined by child classes
        self.param_info: Dict[str, Tuple[float, float]] = {}

    def _compute_critical_density(self) -> float:
        """
        Compute the critical density of the universe.

        Returns
        -------
        float
            Critical density in kg/m³
        """
        H100 = 100 * 1000 / self.Mpc  # 100 km/s/Mpc in SI units
        return 3 * H100**2 / (8 * np.pi * self.G)

    @abstractmethod
    def H(self, z: Union[float, ArrayLike], params: Dict[str, float]) -> Union[float, ArrayLike]:
        """
        Compute the Hubble parameter at redshift z.

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
        """
        pass

    @abstractmethod
    def angular_diameter_distance(self, z: float, params: Dict[str, float]) -> float:
        """
        Compute the angular diameter distance to redshift z.

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
        """
        pass

    def luminosity_distance(self, z: float, params: Dict[str, float]) -> float:
        """
        Compute the luminosity distance to redshift z.

        Parameters
        ----------
        z : float
            Redshift
        params : dict
            Cosmological parameters

        Returns
        -------
        float
            Luminosity distance in Mpc
        """
        return (1 + z) * self.angular_diameter_distance(z, params)

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """
        Validate cosmological parameters against physical bounds.

        Parameters
        ----------
        params : dict
            Cosmological parameters to validate

        Returns
        -------
        bool
            True if parameters are valid, False otherwise

        Raises
        ------
        ValueError
            If required parameters are missing
        """
        required_params = set(self.param_info.keys())
        provided_params = set(params.keys())

        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            raise ValueError(f"Missing required parameters: {missing}")

        # Check if parameters are within physical bounds
        valid = True
        for param, value in params.items():
            if not np.isfinite(value):
                warnings.warn(f"Non-finite value for parameter {param}: {value}")
                valid = False
            if param in self.param_info:
                mean, std = self.param_info[param]
                if abs(value - mean) > 5 * std:  # 5-sigma check
                    warnings.warn(f"Parameter {param} = {
                                  value} is far from expected range")

        return valid

    def get_param_info(self) -> Dict[str, Tuple[float, float]]:
        """
        Get information about the model parameters.

        Returns
        -------
        dict
            Dictionary containing parameter names, fiducial values, and uncertainties
        """
        return self.param_info.copy()

    def __str__(self) -> str:
        """String representation of the cosmological model."""
        return f"{self.__class__.__name__} with parameters: {list(self.param_info.keys())}"

    def __repr__(self) -> str:
        """Detailed string representation of the cosmological model."""
        param_str = ", ".join(f"{k}: ({v[0]:.3f}, {v[1]:.3f})"
                              for k, v in self.param_info.items())
        return f"{self.__class__.__name__}(params={{{param_str}}})"
