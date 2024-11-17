Tutorials
=========

Step-by-step guides to using CMB Analysis.

Getting Started
--------------

.. toctree::
   :maxdepth: 1

   getting_started/installation
   getting_started/first_steps
   getting_started/basic_analysis

Core Concepts
------------

.. toctree::
   :maxdepth: 1

   core/cosmological_models
   core/power_spectra
   core/parameter_estimation
   core/mcmc_analysis

Advanced Topics
-------------

.. toctree::
   :maxdepth: 1

   advanced/custom_models
   advanced/advanced_mcmc
   advanced/error_analysis
   advanced/publication_quality

Best Practices
-------------

.. toctree::
   :maxdepth: 1

   best_practices/numerical_stability
   best_practices/error_handling
   best_practices/performance
   best_practices/visualization

Tutorial: Basic CMB Analysis
--------------------------

This tutorial walks through a basic CMB analysis workflow:

1. Setting up the cosmological model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmb_analysis.cosmology import LCDM
   
   # Initialize model
   model = LCDM()
   
   # Set parameters
   params = {
       'H0': 67.32,
       'omega_b': 0.02237,
       'omega_cdm': 0.1200,
       'tau': 0.0544,
       'ns': 0.9649,
       'ln10As': 3.044
   }

2. Computing power spectra
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmb_analysis.analysis import PowerSpectrumCalculator
   
   # Initialize calculator
   calculator = PowerSpectrumCalculator()
   
   # Compute spectra
   cl_tt, cl_ee, cl_te = calculator.compute_all_spectra(params)

3. Visualizing results
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmb_analysis.visualization import CMBPlotter
   
   # Create plots
   plotter = CMBPlotter()
   plotter.plot_power_spectra({
       'cl_tt': cl_tt,
       'cl_ee': cl_ee,
       'cl_te': cl_te
   })