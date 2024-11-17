"""
Statistical utilities for CMB analysis.
"""

from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats, optimize
import warnings


def compute_covariance(data: ArrayLike,
                       weights: Optional[ArrayLike] = None) -> ArrayLike:
    """
    Compute weighted covariance matrix with error estimation.

    Parameters
    ----------
    data : array-like
        Input data array (n_samples, n_features)
    weights : array-like, optional
        Weights for each sample

    Returns
    -------
    array-like
        Covariance matrix
    """
    data = np.asarray(data)

    if weights is None:
        weights = np.ones(len(data))
    weights = np.asarray(weights)

    # Normalize weights
    weights = weights / np.sum(weights)

    # Weighted mean
    mean = np.average(data, weights=weights, axis=0)

    # Weighted covariance
    dev = data - mean
    cov = np.zeros((data.shape[1], data.shape[1]))

    for i in range(len(data)):
        cov += weights[i] * np.outer(dev[i], dev[i])

    # Bessel correction for weighted samples
    correction = 1 - np.sum(weights**2)
    cov /= correction

    return cov


def chi_square(theory: ArrayLike, data: ArrayLike,
               invcov: Optional[ArrayLike] = None,
               errors: Optional[ArrayLike] = None) -> float:
    """
    Compute χ² statistic with proper error handling.

    Parameters
    ----------
    theory : array-like
        Theoretical predictions
    data : array-like
        Observed data
    invcov : array-like, optional
        Inverse covariance matrix
    errors : array-like, optional
        Individual errors (used if invcov is None)

    Returns
    -------
    float
        χ² value
    """
    residuals = np.asarray(data) - np.asarray(theory)

    if invcov is not None:
        # Full covariance matrix calculation
        try:
            chi2 = residuals @ invcov @ residuals
        except Exception as e:
            warnings.warn(f"Covariance calculation failed: {e}")
            return np.inf
    elif errors is not None:
        # Diagonal errors only
        chi2 = np.sum((residuals / errors)**2)
    else:
        raise ValueError("Must provide either invcov or errors")

    if not np.isfinite(chi2):
        return np.inf

    return chi2


def likelihood_analysis(params: Dict[str, float],
                        model: Callable,
                        data: ArrayLike,
                        errors: ArrayLike,
                        priors: Optional[Dict[str, Tuple[float, float]]] = None
                        ) -> Tuple[float, Dict[str, float]]:
    """
    Perform likelihood analysis for parameter estimation.

    Parameters
    ----------
    params : dict
        Parameter values to evaluate
    model : callable
        Model function
    data : array-like
        Observed data
    errors : array-like
        Observational errors
    priors : dict, optional
        Prior ranges for parameters

    Returns
    -------
    tuple
        (log-likelihood, parameter gradients)
    """
    # Check prior ranges
    if priors is not None:
        for param, value in params.items():
            if param in priors:
                low, high = priors[param]
                if not low <= value <= high:
                    return -np.inf, {p: 0.0 for p in params}

    try:
        # Compute model prediction
        theory = model(**params)

        # Compute chi-square
        chi2 = chi_square(theory, data, errors=errors)

        # Compute log-likelihood
        logL = -0.5 * chi2

        # Compute gradients
        gradients = _compute_gradients(params, model, data, errors)

        return logL, gradients

    except Exception as e:
        warnings.warn(f"Likelihood calculation failed: {e}")
        return -np.inf, {p: 0.0 for p in params}


def _compute_gradients(params: Dict[str, float],
                       model: Callable,
                       data: ArrayLike,
                       errors: ArrayLike,
                       epsilon: float = 1e-7) -> Dict[str, float]:
    """
    Compute numerical gradients of log-likelihood.

    Parameters
    ----------
    params : dict
        Parameter values
    model : callable
        Model function
    data : array-like
        Observed data
    errors : array-like
        Observational errors
    epsilon : float, optional
        Step size for numerical derivatives

    Returns
    -------
    dict
        Parameter gradients
    """
    gradients = {}
    base_theory = model(**params)
    base_chi2 = chi_square(base_theory, data, errors=errors)

    for param in params:
        # Forward step
        params_plus = params.copy()
        params_plus[param] += epsilon
        theory_plus = model(**params_plus)
        chi2_plus = chi_square(theory_plus, data, errors=errors)

        # Backward step
        params_minus = params.copy()
        params_minus[param] -= epsilon
        theory_minus = model(**params_minus)
        chi2_minus = chi_square(theory_minus, data, errors=errors)

        # Central difference gradient
        gradients[param] = (chi2_plus - chi2_minus) / (2 * epsilon)

    return gradients


def parameter_estimation(model: Callable,
                         data: ArrayLike,
                         errors: ArrayLike,
                         param_bounds: Dict[str, Tuple[float, float]],
                         method: str = 'minuit') -> Dict[str, Dict[str, float]]:
    """
    Perform parameter estimation using various methods.

    Parameters
    ----------
    model : callable
        Model function
    data : array-like
        Observed data
    errors : array-like
        Observational errors
    param_bounds : dict
        Parameter bounds
    method : str, optional
        Optimization method ('minuit', 'scipy', 'grid')

    Returns
    -------
    dict
        Parameter estimates with uncertainties
    """
    def chi2_func(params_array):
        params_dict = dict(zip(param_bounds.keys(), params_array))
        theory = model(**params_dict)
        return chi_square(theory, data, errors=errors)

    # Initial parameter guesses (middle of bounds)
    initial_params = np.array([
        (low + high) / 2 for low, high in param_bounds.values()
    ])

    if method == 'minuit':
        try:
            from iminuit import Minuit

            # Setup Minuit
            minuit = Minuit(chi2_func, initial_params)
            minuit.limits = [bounds for bounds in param_bounds.values()]

            # Perform minimization
            minuit.migrad()
            minuit.hesse()

            # Extract results
            results = {}
            for i, param in enumerate(param_bounds.keys()):
                results[param] = {
                    'value': minuit.values[i],
                    'error': minuit.errors[i],
                    'valid': minuit.valid
                }

            return results

        except ImportError:
            warnings.warn("iminuit not available, falling back to scipy")
            method = 'scipy'

    if method == 'scipy':
        try:
            # Perform optimization
            bounds = list(param_bounds.values())
            result = optimize.minimize(chi2_func, initial_params,
                                       method='L-BFGS-B', bounds=bounds)

            # Compute Hessian for errors
            hess_inv = result.hess_inv.todense()

            # Extract results
            results = {}
            for i, param in enumerate(param_bounds.keys()):
                results[param] = {
                    'value': result.x[i],
                    'error': np.sqrt(np.diag(hess_inv)[i]),
                    'valid': result.success
                }

            return results

        except Exception as e:
            warnings.warn(f"Scipy optimization failed: {e}")
            method = 'grid'

    if method == 'grid':
        # Fall back to grid search for robust but slower estimation
        return _grid_search_estimation(model, data, errors, param_bounds)

    raise ValueError(f"Unknown method: {method}")


def _grid_search_estimation(model: Callable,
                            data: ArrayLike,
                            errors: ArrayLike,
                            param_bounds: Dict[str, Tuple[float, float]],
                            n_points: int = 20) -> Dict[str, Dict[str, float]]:
    """
    Perform parameter estimation using grid search.

    Parameters
    ----------
    model : callable
        Model function
    data : array-like
        Observed data
    errors : array-like
        Observational errors
    param_bounds : dict
        Parameter bounds
    n_points : int, optional
        Number of grid points per dimension

    Returns
    -------
    dict
        Parameter estimates with uncertainties
    """
    # Create parameter grids
    param_grids = {
        param: np.linspace(low, high, n_points)
        for param, (low, high) in param_bounds.items()
    }

    # Initialize chi-square grid
    grid_shape = tuple(n_points for _ in param_bounds)
    chi2_grid = np.zeros(grid_shape)

    # Compute chi-square on grid
    for idx in np.ndindex(grid_shape):
        params = {
            param: param_grids[param][i]
            for param, i in zip(param_bounds.keys(), idx)
        }
        theory = model(**params)
        chi2_grid[idx] = chi_square(theory, data, errors=errors)

    # Find best fit
    min_idx = np.unravel_index(np.argmin(chi2_grid), grid_shape)

    # Compute marginalized uncertainties
    results = {}
    for i, (param, grid) in enumerate(param_grids.items()):
        # Get best-fit value
        value = grid[min_idx[i]]

        # Marginalize over other parameters
        axes = tuple(j for j in range(len(param_bounds)) if j != i)
        marg_likelihood = np.exp(-0.5 * np.min(chi2_grid, axis=axes))

        # Normalize
        marg_likelihood /= np.trapz(marg_likelihood, grid)

        # Compute confidence intervals
        cumulative = np.cumsum(marg_likelihood) * np.diff(grid)[0]
        cumulative /= cumulative[-1]

        # Find 1-sigma bounds
        lower = np.interp(0.16, cumulative, grid)
        upper = np.interp(0.84, cumulative, grid)

        results[param] = {
            'value': value,
            'error': (upper - lower) / 2,
            'valid': True
        }

    return results


def compute_correlation_matrix(cov: ArrayLike) -> ArrayLike:
    """
    Compute correlation matrix from covariance matrix.

    Parameters
    ----------
    cov : array-like
        Covariance matrix

    Returns
    -------
    array-like
        Correlation matrix
    """
    cov = np.asarray(cov)
    diag = np.sqrt(np.diag(cov))
    return cov / np.outer(diag, diag)


def fisher_matrix(params: Dict[str, float],
                  model: Callable,
                  errors: ArrayLike,
                  epsilon: float = 1e-7) -> ArrayLike:
    """
    Compute Fisher information matrix.

    Parameters
    ----------
    params : dict
        Parameter values
    model : callable
        Model function
    errors : array-like
        Observational errors
    epsilon : float, optional
        Step size for derivatives

    Returns
    -------
    array-like
        Fisher matrix
    """
    param_names = list(params.keys())
    n_params = len(param_names)
    fisher = np.zeros((n_params, n_params))

    # Compute derivatives
    derivs = []
    base_model = model(**params)

    for param in param_names:
        params_plus = params.copy()
        params_plus[param] += epsilon
        params_minus = params.copy()
        params_minus[param] -= epsilon

        deriv = (model(**params_plus) - model(**params_minus)) / (2 * epsilon)
        derivs.append(deriv)

    # Compute Fisher matrix
    for i in range(n_params):
        for j in range(i + 1):
            fisher[i, j] = np.sum(derivs[i] * derivs[j] / errors**2)
            fisher[j, i] = fisher[i, j]

    return fisher
