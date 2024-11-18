import numpy as np
import matplotlib.pyplot as plt
from cmb_analysis.cosmology import LCDM
from cmb_analysis.analysis import PowerSpectrumCalculator, MCMCAnalysis
from cmb_analysis.visualization import CMBPlotter, MCMCDiagnostics
from cmb_analysis.data import PlanckDataLoader


def main():
    # Set up plotting style
    plt.style.use('seaborn-v0_8-paper')

    # 1. Load Planck Data
    print("Loading Planck data...")
    planck = PlanckDataLoader(data_dir="cmb_analysis/data/planck")

    try:
        # Load theoretical and observed spectra
        theory_data = planck.load_theory_spectra()
        observed_data = planck.load_observed_spectra()

        # Get calibration factor
        cal_factor = planck.get_calibration_factor()
        print(f"Planck calibration factor: {cal_factor}")

        # 2. Compare Theory with Data
        print("\nPreparing theory and data comparison...")
        plotter = CMBPlotter()

        # Prepare data for plotting
        theory = {
            'cl_tt': theory_data['tt'] * cal_factor**2,
            'cl_te': theory_data['te'] * cal_factor**2,
            'cl_ee': theory_data['ee'] * cal_factor**2
        }

        data = {
            'cl_tt': observed_data['tt']['spectrum'],
            'cl_te': observed_data['te']['spectrum'],
            'cl_ee': observed_data['ee']['spectrum']
        }

        errors = {
            'cl_tt': (observed_data['tt']['error_plus'] + observed_data['tt']['error_minus'])/2,
            'cl_te': (observed_data['te']['error_plus'] + observed_data['te']['error_minus'])/2,
            'cl_ee': (observed_data['ee']['error_plus'] + observed_data['ee']['error_minus'])/2
        }

        # Plot comparison
        fig = plotter.plot_power_spectra(theory, data, errors)
        plt.savefig('planck_spectra_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved power spectra comparison plot")

        # 3. Compute and Plot Residuals
        print("\nComputing residuals...")
        fig = plotter.plot_residuals(theory, data, errors)
        plt.savefig('planck_residuals.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved residuals plot")

        # Compute chi-square
        chi2 = planck.compute_chi_square(theory)
        print("\nChi-square values:")
        for spec, value in chi2.items():
            print(f"{spec.upper()}: {value:.2f}")

        # 4. Parameter Estimation
        param_info = {
            'H0': (67.32, 0.54),
            'omega_b': (0.02237, 0.00015),
            'omega_cdm': (0.1200, 0.0012),
            'tau': (0.0544, 0.0073),
            'ns': (0.9649, 0.0042),
            'ln10As': (3.044, 0.014)
        }
        print("\nRunning MCMC analysis...")
        calculator = PowerSpectrumCalculator()
        data_mcmc = {
            'tt_data': observed_data['tt']['spectrum'],
            'tt_error': (observed_data['tt']['error_plus'] + observed_data['tt']['error_minus']) / 2,
            'te_data': observed_data['te']['spectrum'],
            'te_error': (observed_data['te']['error_plus'] + observed_data['te']['error_minus']) / 2,
            'ee_data': observed_data['ee']['spectrum'],
            'ee_error': (observed_data['ee']['error_plus'] + observed_data['ee']['error_minus']) / 2
        }
        mcmc = MCMCAnalysis(power_calculator=calculator,
                            data=data_mcmc, param_info=param_info, n_cores=-1)

        # Run MCMC
        results = mcmc.run_mcmc(progress=True)
        print(type(results))
        # Plot diagnostics
        diagnostics = MCMCDiagnostics()

        # Plot chain evolution
        fig = diagnostics.plot_chain_evolution(
            sampler=results, param_names=list(param_info.keys()))
        plt.savefig('chain_evolution.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved chain evolution plot")

        # Plot autocorrelation
        fig = diagnostics.plot_autocorrelation(results, list(param_info.keys()))
        plt.savefig('autocorrelation.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved autocorrelation plot")

        # Plot convergence metrics
        fig = diagnostics.plot_convergence_metrics(results, list(param_info.keys()))
        plt.savefig('convergence_metrics.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved convergence metrics plot")

        # Plot parameter evolution
        fig = diagnostics.plot_parameter_evolution(results, list(param_info.keys()))
        plt.savefig('parameter_evolution.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved parameter evolution plot")

        # Plot diagnostic summary
        acceptance_fraction = mcmc.compute_convergence_diagnostics()[
            'acceptance_fraction']
        fig = diagnostics.plot_diagnostic_summary(
            results, list(param_info.keys()), acceptance_fraction)
        plt.savefig('diagnostic_summary.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved diagnostic summary plot")

        # Plot parameter constraints
        fig = diagnostics.plot_parameter_constraints(results, list(param_info.keys()))
        plt.savefig('parameter_constraints.png', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved parameter constraints plot")

        # Extract best-fit parameters from MCMC results
        # Reshape to [n_params, n_samples]
        chain = results.coords.T.reshape(len(param_info), -1)
        log_prob = results.log_prob
        best_idx = np.argmax(log_prob)  # Get index of highest likelihood sample
        best_fit_params = {
            name: chain[i, best_idx]
            for i, name in enumerate(param_info.keys())
        }
        print("\nBest-fit parameters:")
        for param, value in best_fit_params.items():
            print(f"{param}: {value:.6f}")

        # Compute best-fit theoretical power spectrum
        cl_tt_fit, cl_ee_fit, cl_te_fit = calculator.compute_all_spectra(
            best_fit_params)

        theory_fit = {
            'cl_tt': cl_tt_fit,
            'cl_te': cl_te_fit,
            'cl_ee': cl_ee_fit
        }

        # Load ACDM image and get axes
        acdm_fig, acdm_ax = plt.subplots(figsize=(10, 8))

        # Plot best-fit theory spectrum with data
        plotter.plot_power_spectra(theory_fit, data, errors)

        # Customize plot
        acdm_ax.set_title('ΛCDM Fit to Planck Data')

        # Save plot
        plt.savefig('acdm_fit.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved ΛCDM fit plot")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
