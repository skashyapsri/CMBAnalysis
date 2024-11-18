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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from functools import partial

from .power_spectrum import PowerSpectrumCalculator


def _worker_init():
    """Initialize worker process to handle numpy parallelization."""
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    np.random.seed()


def _log_probability_worker(theta: ArrayLike, calculator, data_info: Dict) -> float:
    """Standalone worker function for parallel log probability computation."""
    try:
        # Unpack data info
        param_keys = data_info['param_keys']
        bounds = data_info['bounds']
        prior_means = data_info['prior_means']
        prior_stds = data_info['prior_stds']
        data_lengths = data_info['data_lengths']
        data = data_info['data']
        inv_variance = data_info['inv_variance']

        # Check bounds
        params = dict(zip(param_keys, theta))
        if not all(bounds[p][0] <= params[p] <= bounds[p][1] for p in bounds):
            return -np.inf

        # Compute prior
        param_array = np.array([params[p] for p in param_keys])
        chi2_prior = np.sum(((param_array - prior_means) / prior_stds) ** 2)

        # Compute likelihood
        cl_tt, cl_ee, cl_te = calculator.compute_all_spectra(params)

        chi2 = 0.0
        # TT spectrum
        residuals_tt = cl_tt[:data_lengths['tt']] - data['tt_data']
        chi2 += np.sum(residuals_tt**2 * inv_variance['tt'])

        # TE spectrum
        residuals_te = cl_te[:data_lengths['te']] - data['te_data']
        chi2 += np.sum(residuals_te**2 * inv_variance['te'])

        # EE spectrum
        residuals_ee = cl_ee[:data_lengths['ee']] - data['ee_data']
        chi2 += np.sum(residuals_ee**2 * inv_variance['ee'])

        return -0.5 * (chi2 + chi2_prior)

    except Exception as e:
        return -np.inf


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
                 param_info: Dict[str, Tuple[float, float]],
                 n_cores: int = -1) -> None:
        """Initialize MCMC analysis."""
        self.calculator = power_calculator
        self.data = data
        self.param_info = param_info
        self.ndim = len(param_info)

        # Setup parallel processing
        if n_cores == -1:
            self.n_cores = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_cores = min(n_cores, multiprocessing.cpu_count())

        # MCMC settings
        self.nwalkers = max(4 * self.ndim, 2 * self.n_cores)
        self.nsteps = 1000
        self.chunks = max(self.nwalkers // (4 * self.n_cores), 1)

        # Initialize other attributes
        self.sampler = None
        self.chain = None

        # Prepare data for parallel processing
        self._prepare_parallel_data()

    def _setup_pool(self):
        """Setup process pool with proper initialization."""
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.n_cores,
                initializer=_worker_init,
                mp_context=multiprocessing.get_context('fork')
            )
        return self.pool

    def _prepare_parallel_data(self) -> None:
        """Prepare data structure for parallel processing with length matching."""
        print("\nPreparing data for analysis...")

        # Get minimum lengths to ensure consistency
        min_tt_length = min(len(self.data['tt_data']), 2499)  # Theory length is 2499
        min_te_length = min(len(self.data['te_data']), 2499)
        min_ee_length = min(len(self.data['ee_data']), 2499)

        print(f"Using data lengths: TT={min_tt_length}, TE={
              min_te_length}, EE={min_ee_length}")

        # Truncate data to matching lengths
        self.data_lengths = {
            'tt': min_tt_length,
            'te': min_te_length,
            'ee': min_ee_length
        }

        # Normalize data to similar scales
        tt_scale = np.mean(np.abs(self.data['tt_data'][:min_tt_length]))
        te_scale = np.mean(np.abs(self.data['te_data'][:min_te_length]))
        ee_scale = np.mean(np.abs(self.data['ee_data'][:min_ee_length]))

        print(f"\nData scales: TT={tt_scale:.2e}, TE={te_scale:.2e}, EE={ee_scale:.2e}")

        # Pre-compute inverse variances with scaling
        self.inv_variance = {
            'tt': 1.0 / (self.data['tt_error'][:min_tt_length] ** 2),
            'te': 1.0 / (self.data['te_error'][:min_te_length] ** 2),
            'ee': 1.0 / (self.data['ee_error'][:min_ee_length] ** 2)
        }

        # Cache parameter bounds
        self.bounds = {
            'H0': (60.0, 80.0),
            'omega_b': (0.019, 0.025),
            'omega_cdm': (0.10, 0.14),
            'tau': (0.01, 0.10),
            'ns': (0.9, 1.0),
            'ln10As': (2.7, 3.3)
        }

        # Prepare scaled data package for workers
        self.parallel_data = {
            'param_keys': list(self.param_info.keys()),
            'bounds': self.bounds,
            'prior_means': np.array([self.param_info[p][0] for p in self.param_info]),
            'prior_stds': np.array([self.param_info[p][1] for p in self.param_info]),
            'data_lengths': self.data_lengths,
            'data': {
                'tt_data': self.data['tt_data'][:min_tt_length],
                'te_data': self.data['te_data'][:min_te_length],
                'ee_data': self.data['ee_data'][:min_ee_length]
            },
            'inv_variance': self.inv_variance,
            'scales': {
                'tt': tt_scale,
                'te': te_scale,
                'ee': ee_scale
            }
        }

    def _setup_cached_data(self) -> None:
        """Pre-compute and cache frequently used data."""
        # Pre-compute data lengths
        self.data_lengths = {
            'tt': len(self.data['tt_data']),
            'te': len(self.data['te_data']),
            'ee': len(self.data['ee_data'])
        }

        # Pre-compute inverse variances for faster chi-square
        self.inv_variance = {
            'tt': 1.0 / (self.data['tt_error'] ** 2),
            'te': 1.0 / (self.data['te_error'] ** 2),
            'ee': 1.0 / (self.data['ee_error'] ** 2)
        }

        # Cache parameter bounds
        self.bounds = {
            'H0': (60.0, 80.0),
            'omega_b': (0.019, 0.025),
            'omega_cdm': (0.10, 0.14),
            'tau': (0.01, 0.10),
            'ns': (0.9, 1.0),
            'ln10As': (2.7, 3.3)
        }

    def log_prior(self, theta: ArrayLike) -> float:
        """Optimized log prior computation."""
        # Quick bounds check using vectorized operations
        params = dict(zip(self.param_info.keys(), theta))
        if not all(self.bounds[p][0] <= params[p] <= self.bounds[p][1]
                   for p in self.bounds):
            return -np.inf

        # Vectorized chi-square calculation
        param_array = np.array([params[p] for p in self.param_info])
        means = np.array([self.param_info[p][0] for p in self.param_info])
        stds = np.array([self.param_info[p][1] for p in self.param_info])

        chi2 = np.sum(((param_array - means) / stds) ** 2)
        return -0.5 * chi2

    def log_likelihood(self, theta: ArrayLike) -> float:
        """Optimized log likelihood computation."""
        params = dict(zip(self.param_info.keys(), theta))

        try:
            # Compute theoretical spectra
            cl_tt, cl_ee, cl_te = self.calculator.compute_all_spectra(params)

            # Vectorized chi-square calculation
            chi2 = 0.0

            # TT spectrum
            residuals_tt = cl_tt[:self.data_lengths['tt']] - self.data['tt_data']
            chi2 += np.sum(residuals_tt**2 * self.inv_variance['tt'])

            # TE spectrum
            residuals_te = cl_te[:self.data_lengths['te']] - self.data['te_data']
            chi2 += np.sum(residuals_te**2 * self.inv_variance['te'])

            # EE spectrum
            residuals_ee = cl_ee[:self.data_lengths['ee']] - self.data['ee_data']
            chi2 += np.sum(residuals_ee**2 * self.inv_variance['ee'])

            return -0.5 * chi2

        except Exception as e:
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

    def _log_probability_worker(theta: ArrayLike, calculator, data_info: Dict) -> float:
        """Worker function with proper length handling and scaling."""
        try:
            # Unpack data info
            param_keys = data_info['param_keys']
            bounds = data_info['bounds']
            prior_means = data_info['prior_means']
            prior_stds = data_info['prior_stds']
            data_lengths = data_info['data_lengths']
            data = data_info['data']
            inv_variance = data_info['inv_variance']
            scales = data_info['scales']

            # Check bounds
            params = dict(zip(param_keys, theta))
            if not all(bounds[p][0] <= params[p] <= bounds[p][1] for p in bounds):
                return -np.inf

            # Compute prior
            param_array = np.array([params[p] for p in param_keys])
            chi2_prior = np.sum(((param_array - prior_means) / prior_stds) ** 2)

            # Compute theoretical spectra
            try:
                cl_tt, cl_ee, cl_te = calculator.compute_all_spectra(params)
            except Exception as e:
                return -np.inf

            # Compute chi-square with proper length handling and scaling
            chi2 = 0.0

            # TT spectrum
            tt_theory = cl_tt[:data_lengths['tt']]
            tt_data = data['tt_data']
            tt_inv_var = inv_variance['tt']
            chi2_tt = np.sum(((tt_theory - tt_data) ** 2) * tt_inv_var) / scales['tt']

            # TE spectrum
            te_theory = cl_te[:data_lengths['te']]
            te_data = data['te_data']
            te_inv_var = inv_variance['te']
            chi2_te = np.sum(((te_theory - te_data) ** 2) * te_inv_var) / scales['te']

            # EE spectrum
            ee_theory = cl_ee[:data_lengths['ee']]
            ee_data = data['ee_data']
            ee_inv_var = inv_variance['ee']
            chi2_ee = np.sum(((ee_theory - ee_data) ** 2) * ee_inv_var) / scales['ee']

            chi2 = chi2_tt + chi2_te + chi2_ee

            # Return total log probability
            total_logp = -0.5 * (chi2 + chi2_prior)

            if not np.isfinite(total_logp):
                return -np.inf

            return total_logp

        except Exception as e:
            return -np.inf

    def run_mcmc(self, progress: bool = True) -> ArrayLike:
        """Run parallel MCMC analysis with proper initialization."""
        print(f"Starting parallel MCMC analysis with {self.n_cores} cores...")

        try:
            # Initialize walkers
            initial_positions = self._initialize_walkers()
            if initial_positions is None:
                raise ValueError("Failed to initialize walker positions")

            print(f"\nInitialized {self.nwalkers} walkers in {
                  self.ndim}-dimensional space")

            # Create partial function for parallel processing
            log_prob_fn = partial(_log_probability_worker,
                                  calculator=self.calculator,
                                  data_info=self.parallel_data)

            # Setup parallel sampler
            with ProcessPoolExecutor(
                max_workers=self.n_cores,
                initializer=_worker_init,
                mp_context=multiprocessing.get_context('spawn')
            ) as pool:
                # Start with very conservative moves
                moves = [(emcee.moves.GaussianMove(cov=np.eye(self.ndim) * 1e-8), 1.0)]

                self.sampler = emcee.EnsembleSampler(
                    self.nwalkers,
                    self.ndim,
                    log_prob_fn,
                    pool=pool,
                    moves=moves
                )

                # Careful burn-in phase
                if progress:
                    print("\nStarting burn-in phase...")

                # Initial burn-in
                state = self.sampler.run_mcmc(
                    initial_positions,  # Use initial positions directly
                    100,
                    progress=progress
                )

                burn_in_acc = np.mean(self.sampler.acceptance_fraction)
                print(
                    f"Initial burn-in complete. Acceptance fraction: {burn_in_acc:.3f}")

                if burn_in_acc < 0.01:
                    print("Warning: Very low acceptance rate in burn-in")

                self.sampler.reset()

                # Gradually increase step sizes
                spreads = [1e-7, 1e-6, 1e-5]
                for i, spread in enumerate(spreads):
                    moves = [(emcee.moves.GaussianMove(
                        cov=np.eye(self.ndim) * spread), 1.0)]
                    self.sampler.moves = moves

                    state = self.sampler.run_mcmc(
                        state.coords,  # Use the previous state's coordinates
                        100,
                        progress=progress
                    )

                    acc_frac = np.mean(self.sampler.acceptance_fraction)
                    print(f"Burn-in phase {i+1}/{len(spreads)
                                                 } complete. Acceptance: {acc_frac:.3f}")

                    if acc_frac < 0.01:
                        print("Warning: Very low acceptance rate, reducing step size...")
                        spread *= 0.1

                    self.sampler.reset()

                # Production run
                if progress:
                    print("\nStarting production run...")

                # Use a mix of moves for production
                moves = [
                    (emcee.moves.GaussianMove(cov=np.eye(self.ndim) * 1e-5), 0.7),
                    (emcee.moves.DEMove(gamma0=0.5), 0.3)  # Reduced DE step size
                ]
                self.sampler.moves = moves

                # Production run
                self.chain = self.sampler.run_mcmc(
                    state.coords,  # Use final burn-in state
                    self.nsteps,
                    progress=progress
                )

                acc_frac = np.mean(self.sampler.acceptance_fraction)
                print(f"\nMCMC complete! Final acceptance fraction: {acc_frac:.3f}")

                # Additional diagnostics
                log_prob = self.sampler.get_log_prob()
                print(f"Log probability range: [{
                    np.min(log_prob):.2f}, {np.max(log_prob):.2f}]")

                return self.chain

        except Exception as e:
            print(f"\nError during MCMC sampling: {e}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            raise

    def _initialize_walkers(self) -> ArrayLike:
        """Initialize walkers with reliable starting positions."""
        # Use Planck best-fit values
        planck_bestfit = {
            'H0': 67.32,
            'omega_b': 0.02237,
            'omega_cdm': 0.1200,
            'tau': 0.0544,
            'ns': 0.9649,
            'ln10As': 3.044
        }

        # Initialize from best-fit
        means = np.array([planck_bestfit[p] for p in self.param_info.keys()])
        stds = np.array([self.param_info[p][1] for p in self.param_info])

        print("\nInitializing walkers around Planck best-fit values:")
        for name, value in zip(self.param_info.keys(), means):
            print(f"{name}: {value}")

        # Initialize walkers in a small ball around best-fit
        initial_spread = 1e-5
        initial = np.zeros((self.nwalkers, self.ndim))

        print("\nGenerating initial walker positions...")

        # Ensure we get enough valid positions
        valid_positions = 0
        max_attempts = 10000
        attempts = 0

        while valid_positions < self.nwalkers and attempts < max_attempts:
            # Generate position
            pos = means + stds * initial_spread * np.random.randn(self.ndim)

            # Check if position is valid
            if self._check_parameter_bounds(dict(zip(self.param_info.keys(), pos))):
                logp = _log_probability_worker(pos, self.calculator, self.parallel_data)
                if np.isfinite(logp):
                    initial[valid_positions] = pos
                    valid_positions += 1
                    if valid_positions % 10 == 0:
                        print(f"Initialized {valid_positions}/{self.nwalkers} walkers")

            attempts += 1

            # Adjust spread if having trouble
            if attempts % 1000 == 0:
                initial_spread *= 0.5
                print(f"Adjusting spread to {initial_spread}")

        if valid_positions < self.nwalkers:
            raise ValueError(f"Could only initialize {
                valid_positions}/{self.nwalkers} walkers")

        # Verify positions
        log_probs = np.array([
            _log_probability_worker(pos, self.calculator, self.parallel_data)
            for pos in initial
        ])

        finite_probs = np.isfinite(log_probs)
        if not np.all(finite_probs):
            print(f"Warning: {np.sum(~finite_probs)
                              } walkers have invalid initial positions")

        print(f"\nInitial log probability range: [{np.min(log_probs[finite_probs]):.2f}, {
            np.max(log_probs[finite_probs]):.2f}]")

        return initial

    def __del__(self):
        """Ensure proper cleanup of parallel resources."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = Non

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
        """Get MCMC chain with optimized memory usage."""
        if self.sampler is None:
            raise ValueError("No MCMC chain available. Run MCMC first.")

        # Use memory-efficient slicing
        chain = self.sampler.get_chain(discard=discard, thin=thin, flat=flat)
        if flat:
            return chain
        return chain.copy()

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
