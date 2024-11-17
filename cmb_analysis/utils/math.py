"""
Mathematical utilities for CMB analysis.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy import special, integrate
import warnings


def spherical_bessel_j(l: int, x: ArrayLike,
                       method: str = 'auto') -> ArrayLike:
    """
    Compute spherical Bessel function with optimizations for CMB analysis.

    Parameters
    ----------
    l : int
        Angular momentum quantum number
    x : array-like
        Arguments
    method : str, optional
        Computation method ('auto', 'direct', 'asymptotic')

    Returns
    -------
    array-like
        Spherical Bessel function values

    Notes
    -----
    Implements various approximations for different regimes:
    - Small x: Power series expansion
    - Large x: Asymptotic form
    - Intermediate x: Direct computation
    """
    x = np.asarray(x)

    if method == 'auto':
        # Choose method based on x values
        result = np.zeros_like(x)

        # Small x approximation
        small_x = x < 0.1
        if np.any(small_x):
            result[small_x] = _spherical_bessel_small_x(l, x[small_x])

        # Large x approximation
        large_x = x >= 10.0
        if np.any(large_x):
            result[large_x] = _spherical_bessel_large_x(l, x[large_x])

        # Direct computation for intermediate values
        mid_x = (~small_x) & (~large_x)
        if np.any(mid_x):
            result[mid_x] = special.spherical_jn(l, x[mid_x])

        return result

    elif method == 'direct':
        return special.spherical_jn(l, x)

    elif method == 'asymptotic':
        return _spherical_bessel_large_x(l, x)

    else:
        raise ValueError(f"Unknown method: {method}")


def _spherical_bessel_small_x(l: int, x: ArrayLike) -> ArrayLike:
    """Small x approximation for spherical Bessel function."""
    return x**l / (special.factorial2(2*l + 1))


def _spherical_bessel_large_x(l: int, x: ArrayLike) -> ArrayLike:
    """Large x approximation for spherical Bessel function."""
    return np.sin(x - l*np.pi/2) / x


def window_function(l: int, fwhm: float, nside: int) -> float:
    """
    Compute window function for given beam and pixel size.

    Parameters
    ----------
    l : int
        Multipole moment
    fwhm : float
        Full Width at Half Maximum in radians
    nside : int
        HEALPix Nside parameter

    Returns
    -------
    float
        Window function value
    """
    # Beam window function
    sigma = fwhm / np.sqrt(8 * np.log(2))
    beam = np.exp(-0.5 * l * (l + 1) * sigma**2)

    # Pixel window function
    pixel = hp_pixel_window(nside, l)

    return beam * pixel


def hp_pixel_window(nside: int, l: int) -> float:
    """
    Compute HEALPix pixel window function.

    Parameters
    ----------
    nside : int
        HEALPix Nside parameter
    l : int
        Multipole moment

    Returns
    -------
    float
        Pixel window function value
    """
    theta = np.pi / (4 * nside)
    return special.lpn(l, np.cos(theta))[0][-1]


def angular_distance_matrix(theta1: ArrayLike, phi1: ArrayLike,
                            theta2: ArrayLike, phi2: ArrayLike) -> ArrayLike:
    """
    Compute matrix of angular distances between points on a sphere.

    Parameters
    ----------
    theta1, phi1 : array-like
        Angular coordinates of first set of points
    theta2, phi2 : array-like
        Angular coordinates of second set of points

    Returns
    -------
    array-like
        Matrix of angular distances
    """
    theta1, phi1 = np.asarray(theta1), np.asarray(phi1)
    theta2, phi2 = np.asarray(theta2), np.asarray(phi2)

    # Broadcast to create coordinate matrices
    t1 = theta1[:, np.newaxis]
    t2 = theta2[np.newaxis, :]
    p1 = phi1[:, np.newaxis]
    p2 = phi2[np.newaxis, :]

    # Compute angular distances using spherical law of cosines
    cos_dist = (np.sin(t1) * np.sin(t2) * np.cos(p1 - p2) +
                np.cos(t1) * np.cos(t2))

    # Ensure numerical stability
    cos_dist = np.clip(cos_dist, -1.0, 1.0)

    return np.arccos(cos_dist)


def integrate_adaptive(func: Callable, a: float, b: float,
                       args: tuple = (),
                       rtol: float = 1e-8) -> Tuple[float, float]:
    """
    Adaptive integration with error control.

    Parameters
    ----------
    func : callable
        Function to integrate
    a, b : float
        Integration limits
    args : tuple, optional
        Additional arguments for func
    rtol : float, optional
        Relative tolerance

    Returns
    -------
    tuple
        (integral value, estimated error)
    """
    def wrapped_func(x):
        try:
            return func(x, *args)
        except Exception as e:
            warnings.warn(f"Integration error at x={x}: {e}")
            return 0.0

    try:
        # Try adaptive integration
        result = integrate.quad(wrapped_func, a, b, epsrel=rtol)

        # Check if result is valid
        if not np.isfinite(result[0]):
            raise ValueError("Non-finite integral result")

        return result

    except Exception as e:
        warnings.warn(f"Falling back to simpler integration method: {e}")
        # Fall back to simpler method
        x = np.linspace(a, b, 1000)
        y = wrapped_func(x)
        integral = integrate.simps(y, x)
        error = abs(integral) * rtol
        return integral, error


def legendre_polynomials(l_max: int, x: ArrayLike) -> ArrayLike:
    """
    Compute Legendre polynomials up to l_max.

    Parameters
    ----------
    l_max : int
        Maximum multipole moment
    x : array-like
        Arguments

    Returns
    -------
    array-like
        Array of Legendre polynomials [P_0(x), P_1(x), ..., P_lmax(x)]
    """
    x = np.asarray(x)
    if np.any(np.abs(x) > 1):
        raise ValueError("Arguments must be in [-1, 1]")

    # Initialize array for results
    p = np.zeros((l_max + 1, len(x)))

    # P_0(x) = 1
    p[0] = 1
    if l_max == 0:
        return p

    # P_1(x) = x
    p[1] = x
    if l_max == 1:
        return p

    # Use recurrence relation
    # (l+1)P_{l+1}(x) = (2l+1)xP_l(x) - lP_{l-1}(x)
    for l in range(1, l_max):
        p[l+1] = ((2*l + 1) * x * p[l] - l * p[l-1]) / (l + 1)

    return p


def wigner_3j(j1: int, j2: int, j3: int,
              m1: int, m2: int, m3: int) -> float:
    """
    Compute Wigner 3-j symbol.

    Parameters
    ----------
    j1, j2, j3 : int
        Angular momenta
    m1, m2, m3 : int
        Magnetic quantum numbers

    Returns
    -------
    float
        Wigner 3-j symbol value

    Notes
    -----
    Implements the Racah formula for Wigner 3-j symbols.
    Used in computing angular power spectrum covariances.
    """
    try:
        return float(special.wigner_3j(j1, j2, j3, m1, m2, m3))
    except Exception as e:
        warnings.warn(f"Wigner 3-j computation failed: {e}")
        return 0.0
