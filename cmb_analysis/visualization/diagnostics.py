"""
MCMC diagnostic tools and visualizations.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class MCMCDiagnostics:
    """
    Diagnostic tools for MCMC analysis.

    This class provides methods for:
    - Chain convergence assessment
    - Parameter evolution visualization
    - Autocorrelation analysis
    - Chain statistics
    """

    def __init__(self) -> None:
        """Initialize diagnostic tools."""
        self.setup_style()

    def setup_style(self) -> None:
        """Set up plotting style for diagnostics."""
        plt.style.use('seaborn-v0_8-paper')
        self.colors = sns.color_palette("deep")

    def plot_chain_evolution(self, chain: ArrayLike,
                             param_names: List[str]) -> Figure:
        """
        Plot MCMC chain evolution.

        Parameters
        ----------
        chain : array-like
            MCMC chain with shape (steps, walkers, parameters)
        param_names : list
            Parameter names

        Returns
        -------
        Figure
            Chain evolution plot
        """
        n_params = chain.shape[2]
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params),
                                 sharex=True)

        if n_params == 1:
            axes = [axes]

        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.plot(chain[:, :, i], alpha=0.3)
            ax.set_ylabel(name)
            ax.axhline(np.mean(chain[:, :, i]),
                       color='r', linestyle='--')

        axes[-1].set_xlabel('Step number')
        plt.tight_layout()
        return fig

    def plot_autocorrelation(self, chain: ArrayLike,
                             param_names: List[str]) -> Figure:
        """
        Plot autocorrelation functions.

        Parameters
        ----------
        chain : array-like
            MCMC chain
        param_names : list
            Parameter names

        Returns
        -------
        Figure
            Autocorrelation plot
        """
        n_params = chain.shape[2]
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params),
                                 sharex=True)

        if n_params == 1:
            axes = [axes]

        for i, (ax, name) in enumerate(zip(axes, param_names)):
            flat_chain = chain[:, :, i].flatten()
            acf = self._compute_acf(flat_chain)
            ax.plot(acf)
            ax.set_ylabel(f'ACF ({name})')
            ax.axhline(0, color='k', linestyle='--')

        axes[-1].set_xlabel('Lag')
        plt.tight_layout()
        return fig

    def _compute_acf(self, x: ArrayLike) -> ArrayLike:
        """
        Compute autocorrelation function.

        Parameters
        ----------
        x : array-like
            Input time series

        Returns
        -------
        array-like
            Autocorrelation function
        """
        mean = np.mean(x)
        var = np.var(x)
        xp = x - mean
        corr = np.correlate(xp, xp, mode='full')[len(x)-1:]
        return corr / var / len(x)

    def plot_convergence_metrics(self, chain: ArrayLike,
                                 param_names: List[str]) -> Figure:
        """
        Plot convergence diagnostics.

        Parameters
        ----------
        chain : array-like
            MCMC chain
        param_names : list
            Parameter names

        Returns
        -------
        Figure
            Convergence diagnostics plot
        """
        # Compute Gelman-Rubin statistics
        gr_stats = self.gelman_rubin_statistic(chain)
        n_eff = self.effective_sample_size(chain)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot Gelman-Rubin statistics
        axes[0].axhline(y=1.1, color='r', linestyle='--',
                        label='GR threshold')
        axes[0].bar(param_names, gr_stats, alpha=0.6)
        axes[0].set_ylabel('Gelman-Rubin statistic')
        axes[0].legend()

        # Plot effective sample sizes
        axes[1].bar(param_names, n_eff, alpha=0.6)
        axes[1].set_ylabel('Effective sample size')

        # Rotate x-axis labels if needed
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def gelman_rubin_statistic(self, chain: ArrayLike) -> ArrayLike:
        """
        Compute Gelman-Rubin statistics.

        Parameters
        ----------
        chain : array-like
            MCMC chain

        Returns
        -------
        array-like
            Gelman-Rubin statistics for each parameter
        """
        n_steps, n_walkers, n_params = chain.shape

        # Use second half of chain
        chain = chain[n_steps//2:]
        n_steps = len(chain)

        gr_stats = np.zeros(n_params)

        for i in range(n_params):
            chain_param = chain[:, :, i]

            # Between-chain variance
            B = n_steps * np.var(np.mean(chain_param, axis=0))

            # Within-chain variance
            W = np.mean(np.var(chain_param, axis=0))

            # Estimate of marginal posterior variance
            V = ((n_steps - 1) * W + B) / n_steps

            # R-hat statistic
            gr_stats[i] = np.sqrt(V / W)

        return gr_stats

    def effective_sample_size(self, chain: ArrayLike) -> ArrayLike:
        """
        Compute effective sample size.

        Parameters
        ----------
        chain : array-like
            MCMC chain

        Returns
        -------
        array-like
            Effective sample sizes for each parameter
        """
        n_steps, n_walkers, n_params = chain.shape

        # Flatten chain across walkers
        flat_chain = chain.reshape(-1, n_params)

        n_eff = np.zeros(n_params)

        for i in range(n_params):
            acf = self._compute_acf(flat_chain[:, i])

            # Find first negative ACF value
            neg_idx = np.where(acf < 0)[0]
            max_idx = neg_idx[0] if len(neg_idx) > 0 else len(acf)

            # Compute integrated autocorrelation time
            tau = 1 + 2 * np.sum(acf[1:max_idx])

            # Compute effective sample size
            n_eff[i] = len(flat_chain) / tau

        return n_eff

    def plot_parameter_evolution(self, chain: ArrayLike,
                                 param_names: List[str],
                                 true_values: Optional[ArrayLike] = None) -> Figure:
        """
        Plot detailed parameter evolution.

        Parameters
        ----------
        chain : array-like
            MCMC chain
        param_names : list
            Parameter names
        true_values : array-like, optional
            True parameter values

        Returns
        -------
        Figure
            Parameter evolution plot
        """
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 2, figsize=(15, 4*n_params))

        for i, (name, true_val) in enumerate(zip(param_names,
                                                 true_values if true_values is not None
                                                 else [None]*n_params)):
            # Chain evolution
            axes[i, 0].plot(chain[:, :, i], alpha=0.3)
            if true_val is not None:
                axes[i, 0].axhline(true_val, color='r',
                                   linestyle='--', label='True')
            axes[i, 0].set_ylabel(name)
            axes[i, 0].set_xlabel('Step')

            # Running mean
            mean = np.mean(chain[:, :, i], axis=1)
            std = np.std(chain[:, :, i], axis=1)
            axes[i, 1].plot(mean, label='Mean')
            axes[i, 1].fill_between(np.arange(len(mean)),
                                    mean - std, mean + std,
                                    alpha=0.3)
            if true_val is not None:
                axes[i, 1].axhline(true_val, color='r',
                                   linestyle='--', label='True')
            axes[i, 1].set_ylabel(name)
            axes[i, 1].set_xlabel('Step')
            axes[i, 1].legend()

        plt.tight_layout()
        return fig

    def plot_diagnostic_summary(self, chain: ArrayLike,
                                param_names: List[str],
                                acceptance_fraction: float) -> Figure:
        """
        Create comprehensive diagnostic summary plot.

        Parameters
        ----------
        chain : array-like
            MCMC chain
        param_names : list
            Parameter names
        acceptance_fraction : float
            MCMC acceptance fraction

        Returns
        -------
        Figure
            Diagnostic summary plot
        """
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2)

        # Chain evolution
        ax1 = fig.add_subplot(gs[0, :])
        for i, name in enumerate(param_names):
            ax1.plot(chain[:, :, i].mean(axis=1),
                     label=name, alpha=0.7)
        ax1.set_title('Parameter Evolution')
        ax1.set_xlabel('Step')
        ax1.legend()

        # Gelman-Rubin statistics
        ax2 = fig.add_subplot(gs[1, 0])
        gr_stats = self.gelman_rubin_statistic(chain)
        ax2.bar(param_names, gr_stats, alpha=0.6)
        ax2.axhline(y=1.1, color='r', linestyle='--',
                    label='Threshold')
        ax2.set_title('Gelman-Rubin Statistics')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()

        # Effective sample size
        ax3 = fig.add_subplot(gs[1, 1])
        n_eff = self.effective_sample_size(chain)
        ax3.bar(param_names, n_eff, alpha=0.6)
        ax3.set_title('Effective Sample Size')
        ax3.tick_params(axis='x', rotation=45)

        # Summary statistics
        ax4 = fig.add_subplot(gs[2, :])
        summary_text = [
            f"Chain length: {len(chain)}",
            f"Number of walkers: {chain.shape[1]}",
            f"Acceptance fraction: {acceptance_fraction:.3f}",
            "\nConvergence Metrics:",
            "Gelman-Rubin (should be < 1.1):"
        ]

        for name, gr in zip(param_names, gr_stats):
            summary_text.append(f"  {name}: {gr:.3f}")

        ax4.text(0.05, 0.95, '\n'.join(summary_text),
                 transform=ax4.transAxes,
                 verticalalignment='top',
                 fontfamily='monospace')
        ax4.set_axis_off()

        plt.tight_layout()
        return fig
