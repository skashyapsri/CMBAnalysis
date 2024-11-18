import warnings
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
"""
Planck Legacy Archive data loader and handler.
Handles PR3 (2018) power spectra data.
"""


class PlanckDataLoader:
    """
    Handler for Planck Legacy Archive CMB power spectra data.
    """

    def __init__(self, data_dir: str = "data/planck") -> None:
        """
        Initialize Planck data loader.

        Parameters
        ----------
        data_dir : str
            Directory containing Planck data files
        """
        self.data_dir = Path(data_dir)
        self.theory_data = None
        self.tt_data = None
        self.te_data = None
        self.ee_data = None

    def load_theory_spectra(self) -> Dict[str, np.ndarray]:
        """
        Load theoretical power spectra from Planck best-fit model.

        Returns
        -------
        dict
            Dictionary containing 'ell', 'tt', 'te', 'ee', 'bb', 'pp' arrays
        """
        filename = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
        try:
            data = np.loadtxt(self.data_dir / filename, skiprows=1)
            self.theory_data = {
                'ell': data[:, 0],
                'tt': data[:, 1],
                'te': data[:, 2],
                'ee': data[:, 3],
                'bb': data[:, 4],
                'pp': data[:, 5]
            }
            return self.theory_data
        except Exception as e:
            raise IOError(f"Error loading theory spectra: {e}")

    def load_observed_spectra(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load observed power spectra with error bars.

        Returns
        -------
        dict
            Dictionary containing TT, TE, EE spectra with errors
        """
        try:
            # Load TT spectrum
            tt_file = "COM_PowerSpect_CMB-TT-full_R3.01.txt"
            tt_data = np.loadtxt(self.data_dir / tt_file, skiprows=1)
            self.tt_data = {
                'ell': tt_data[:, 0],
                'spectrum': tt_data[:, 1],
                'error_minus': tt_data[:, 2],
                'error_plus': tt_data[:, 3]
            }

            # Load TE spectrum
            te_file = "COM_PowerSpect_CMB-TE-full_R3.01.txt"
            te_data = np.loadtxt(self.data_dir / te_file, skiprows=1)
            self.te_data = {
                'ell': te_data[:, 0],
                'spectrum': te_data[:, 1],
                'error_minus': te_data[:, 2],
                'error_plus': te_data[:, 3]
            }

            # Load EE spectrum
            ee_file = "COM_PowerSpect_CMB-EE-full_R3.01.txt"
            ee_data = np.loadtxt(self.data_dir / ee_file, skiprows=1)
            self.ee_data = {
                'ell': ee_data[:, 0],
                'spectrum': ee_data[:, 1],
                'error_minus': ee_data[:, 2],
                'error_plus': ee_data[:, 3]
            }

            return {
                'tt': self.tt_data,
                'te': self.te_data,
                'ee': self.ee_data
            }

        except Exception as e:
            raise IOError(f"Error loading observed spectra: {e}")

    def get_binned_spectra(self) -> Dict[str, np.ndarray]:
        """
        Get binned power spectra for faster analysis.

        Returns
        -------
        dict
            Dictionary containing binned spectra
        """
        if self.tt_data is None:
            self.load_observed_spectra()

        def bin_spectrum(data: Dict[str, np.ndarray],
                         bin_size: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Bin spectrum data with errors."""
            n_bins = len(data['ell']) // bin_size
            ell_binned = np.zeros(n_bins)
            cl_binned = np.zeros(n_bins)
            error_binned = np.zeros(n_bins)

            for i in range(n_bins):
                idx_start = i * bin_size
                idx_end = (i + 1) * bin_size
                ell_binned[i] = np.mean(data['ell'][idx_start:idx_end])
                cl_binned[i] = np.mean(data['spectrum'][idx_start:idx_end])
                # Combine errors in quadrature
                error_binned[i] = np.sqrt(np.mean(
                    data['error_plus'][idx_start:idx_end]**2
                ))

            return ell_binned, cl_binned, error_binned

        return {
            'tt': bin_spectrum(self.tt_data),
            'te': bin_spectrum(self.te_data),
            'ee': bin_spectrum(self.ee_data)
        }

    def compute_chi_square(self, theory_cls: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute χ² between theory and data.

        Parameters
        ----------
        theory_cls : dict
            Dictionary containing theoretical Cl values

        Returns
        -------
        dict
            χ² values for each spectrum type
        """
        def calc_chi2(data: Dict[str, np.ndarray],
                      theory: np.ndarray) -> float:
            """Calculate χ² for a single spectrum."""
            # Truncate theory spectrum to match data length
            theory_truncated = theory[:len(data['spectrum'])]
            residuals = data['spectrum'] - theory_truncated
            errors = (data['error_plus'] + data['error_minus']) / 2
            return np.sum((residuals / errors)**2)

        return {
            'tt': calc_chi2(self.tt_data, theory_cls['cl_tt']),
            'te': calc_chi2(self.te_data, theory_cls['cl_te']),
            'ee': calc_chi2(self.ee_data, theory_cls['cl_ee'])
        }

    def get_calibration_factor(self) -> float:
        """
        Get Planck calibration factor.

        Returns
        -------
        float
            Calibration factor to apply to theory spectra
        """
        try:
            params_file = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt"
            with open(self.data_dir / params_file, 'r') as f:
                for line in f:
                    if 'calPlanck' in line:
                        return float(line.split()[-1])
        except Exception:
            warnings.warn(
                "Could not load calibration factor, using default 0.1000442E+01")
            return 0.1000442E+01
