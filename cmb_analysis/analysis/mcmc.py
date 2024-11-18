"""
Implementation of MCMC analysis for CMB parameter estimation.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from numpy.typing import ArrayLike
import emcee
import corner
from scipy import stats
import warnings

from .power_spectrum import PowerSpectrumCalculator


class MCMCAnalysis:
    """
    MCMC-based parameter estimation for CMB analysis.

    This class handles:
    - MCMC sampling of parameter space
    - Convergence diagnostics
    - Parameter constraints
    - Chain analysis and visualization
    """

    def __init__(self, power_calculator: PowerSpectrumCalculator,
                 data: Dict[str, ArrayLike],
                 param_info: Dict[str, Tuple[float, float]]) -> None:
        """
        Initialize MCMC analysis.

        Parameters
        ----------
        power_calculator : PowerSpectrumCalculator
            Calculator for theoretical power spectra
        data : dict
            Dictionary containing observed spectra and errors
        param_info : dict
            Parameter information (means and std. devs)
        """
        self.calculator = power_calculator
        self.data = data
        self.param_info = param_info

        # MCMC settings
        self.nwalkers = 64
        self.nsteps = 1000
        self.ndim = len(param_info)

        # Initialize sampler
        self.sampler = None
        self.chain = None

    def log_prior(self, theta: ArrayLike) -> float:
        """
        Compute log prior probability.

        Parameters
        ----------
        theta : array-like
            Parameter vector

        Returns
        -------
        float
            Log prior probability
        """
        params = dict(zip(self.param_info.keys(), theta))

        # Check physical bounds
        if not self._check_parameter_bounds(params):
            return -np.inf

        # Gaussian priors
        chi2 = 0
        for param, (mean, std) in self.param_info.items():
            chi2 += ((params[param] - mean) / std)**2

        return -0.5 * chi2

    def log_likelihood(self, theta: ArrayLike) -> float:
        """
        Compute log likelihood.

        Parameters
        ----------
        theta : array-like
            Parameter vector

        Returns
        -------
        float
            Log likelihood
        """
        params = dict(zip(self.param_info.keys(), theta))

        try:
            # Compute theoretical spectra
            cl_tt, cl_ee, cl_te = self.calculator.compute_all_spectra(params)

            # Truncate theoretical spectra to match observed data length
            cl_tt = cl_tt[:len(self.data['tt_data'])]
            cl_te = cl_te[:len(self.data['te_data'])]
            cl_ee = cl_ee[:len(self.data['ee_data'])]

            # Compute chi-square
            chi2_tt = self.calculator.compute_chi_square(
                cl_tt, self.data['tt_data'], self.data['tt_error']
            )
            chi2_te = self.calculator.compute_chi_square(
                cl_te, self.data['te_data'], self.data['te_error']
            )
            chi2_ee = self.calculator.compute_chi_square(
                cl_ee, self.data['ee_data'], self.data['ee_error']
            )

            return -0.5 * (chi2_tt + chi2_te + chi2_ee)

        except Exception as e:
            warnings.warn(f"Error in likelihood calculation: {e}")
            return -np.inf

    def log_probability(self, theta: ArrayLike) -> float:
        """
        Compute total log probability.

        Parameters
        ----------
        theta : array-like
            Parameter vector

        Returns
        -------
        float
            Log probability
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_mcmc(self, progress: bool = True) -> ArrayLike:
        """
        Run MCMC analysis.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress bar

        Returns
        -------
        array-like
            MCMC chain with shape (steps, walkers, parameters)
        """
        print("Starting MCMC analysis...")

        # Initialize walkers
        initial_positions = self._initialize_walkers()

        # Set up sampler
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.log_probability
        )

        # Run MCMC
        try:
            self.chain = self.sampler.run_mcmc(
                initial_positions, self.nsteps,
                progress=progress
            )
            print("MCMC complete!")
            return self.chain
        except Exception as e:
            print(f"Error during MCMC sampling: {e}")
            raise

    def _initialize_walkers(self) -> ArrayLike:
        """
        Initialize walker positions.

        Returns
        -------
        array-like
            Initial walker positions
        """
        initial = np.zeros((self.nwalkers, self.ndim))
        for i, (param, (mean, std)) in enumerate(self.param_info.items()):
            initial[:, i] = mean + std * np.random.randn(self.nwalkers) * 0.1
        return initial

    def _check_parameter_bounds(self, params: Dict[str, float]) -> bool:
        """
        Check if parameters are within physical bounds.

        Parameters
        ----------
        params : dict
            Parameter dictionary

        Returns
        -------
        bool
            Whether parameters are valid
        """
        bounds = {
            'H0': (60.0, 80.0),
            'omega_b': (0.019, 0.025),
            'omega_cdm': (0.10, 0.14),
            'tau': (0.01, 0.10),
            'ns': (0.9, 1.0),
            'ln10As': (2.7, 3.3)
        }

        for param, value in params.items():
            if param in bounds:
                if not bounds[param][0] <= value <= bounds[param][1]:
                    return False
        return True

    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False) -> ArrayLike:
        """
        Get MCMC chain.

        Parameters
        ----------
        discard : int, optional
            Number of steps to discard
        thin : int, optional
            Thinning factor
        flat : bool, optional
            Whether to flatten the chain

        Returns
        -------
        array-like
            MCMC chain
        """
        if self.sampler is None:
            raise ValueError("No MCMC chain available. Run MCMC first.")
        return self.sampler.get_chain(discard=discard, thin=thin, flat=flat)

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics from MCMC chain.

        Returns
        -------
        dict
            Dictionary containing parameter statistics
        """
        flat_samples = self.get_chain(discard=100, thin=15, flat=True)
        stats = {}

        for i, param in enumerate(self.param_info.keys()):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            stats[param] = {
                'mean': np.mean(flat_samples[:, i]),
                'std': np.std(flat_samples[:, i]),
                'median': mcmc[1],
                'lower': mcmc[1] - mcmc[0],
                'upper': mcmc[2] - mcmc[1]
            }

        return stats

    def get_best_fit(self) -> Dict[str, float]:
        """
        Get best-fit parameters.

        Returns
        -------
        dict
            Best-fit parameter values
        """
        flat_samples = self.get_chain(flat=True)
        best_idx = np.argmax(self.sampler.get_log_prob(flat=True))

        return dict(zip(
            self.param_info.keys(),
            flat_samples[best_idx]
        ))

    def compute_convergence_diagnostics(self) -> Dict[str, float]:
        """
        Compute MCMC convergence diagnostics.

        Returns
        -------
        dict
            Dictionary containing convergence metrics
        """
        chain = self.get_chain()

        # Compute Gelman-Rubin statistic
        n_steps, n_walkers, n_params = chain.shape
        gr_stats = []

        for i in range(n_params):
            chain_param = chain[:, :, i]
            B = n_steps * np.var(np.mean(chain_param, axis=0))
            W = np.mean(np.var(chain_param, axis=0))
            V = ((n_steps - 1) * W + B) / n_steps
            R = np.sqrt(V / W)
            gr_stats.append(R)

        # Compute effective sample size
        n_eff = []
        flat_chain = chain.reshape(-1, n_params)

        for i in range(n_params):
            acf = self._compute_autocorr(flat_chain[:, i])
            tau = 1 + 2 * np.sum(acf[1:])
            n_eff.append(len(flat_chain) / tau)

        return {
            'gelman_rubin': dict(zip(self.param_info.keys(), gr_stats)),
            'effective_samples': dict(zip(self.param_info.keys(), n_eff)),
            'acceptance_fraction': np.mean(self.sampler.acceptance_fraction)
        }

    def _compute_autocorr(self, x: ArrayLike) -> ArrayLike:
        """
        Compute autocorrelation function.

        Parameters
        ----------
        x : array-like
            Input array

        Returns
        -------
        array-like
            Autocorrelation function
        """
        n = len(x)
        x = x - np.mean(x)
        result = np.correlate(x, x, mode='full')
        result = result[n-1:]
        return result[:n//3] / result[0]
