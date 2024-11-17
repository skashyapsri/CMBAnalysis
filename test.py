import numpy as np
from cmb_analysis.cosmology import LCDM
from cmb_analysis.analysis import PowerSpectrumCalculator
from cmb_analysis.visualization import CMBPlotter

# Step 1: Load the data from the file
data_file = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
data = np.loadtxt(data_file, skiprows=1)

# Data columns
l = data[:, 0]       # Multipole moments (ell)
Dl_tt = data[:, 1]   # TT Power spectrum values
Dl_te = data[:, 2]   # TE Power spectrum values
Dl_ee = data[:, 3]   # EE Power spectrum values
# Error columns (for now, we assume that the error format applies to all spectra)
dDl_tt_min = data[:, 4]  # Negative error for TT
dDl_tt_plus = data[:, 5]  # Positive error for TT

# Step 2: Prepare the observed data and errors
data_dict = {
    'cl_tt': Dl_tt,
    'cl_te': Dl_te,
    'cl_ee': Dl_ee
}
errors_dict = {
    'cl_tt': (np.abs(dDl_tt_min) + np.abs(dDl_tt_plus)) / 2,
    # Assuming TE errors are similar
    'cl_te': (np.abs(dDl_tt_min) + np.abs(dDl_tt_plus)) / 2,
    # Assuming EE errors are similar
    'cl_ee': (np.abs(dDl_tt_min) + np.abs(dDl_tt_plus)) / 2
}

# Step 3: Initialize the LCDM model
model = LCDM()
params = {
    'H0': 67.32,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'tau': 0.0544,
    'ns': 0.9649,
    'ln10As': 3.044
}

# Step 4: Compute theoretical power spectra
calculator = PowerSpectrumCalculator()
cl_tt, cl_ee, cl_te = calculator.compute_all_spectra(params)

# Prepare theory dictionary with the spectra
theory_dict = {
    'cl_tt': cl_tt,
    'cl_ee': cl_ee,
    'cl_te': cl_te
}
print(theory_dict)
# Step 5: Plot the power spectra
plotter = CMBPlotter()
fig = plotter.plot_power_spectra(theory=theory_dict, data=data_dict, errors=errors_dict)

# Optional: Save the plot
plotter.save_publication_plots("cmb_power_spectra", fig)
