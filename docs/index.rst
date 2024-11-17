CMB Analysis Documentation
=========================

A comprehensive framework for analyzing Cosmic Microwave Background radiation data.

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quick_start
   overview

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index
   examples/index
   theory/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cosmology
   api/analysis
   api/visualization
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   roadmap

Features
--------

- Modern cosmological model implementations (ΛCDM, wCDM)
- Advanced CMB power spectrum analysis
- MCMC parameter estimation
- Publication-quality visualization
- Comprehensive error analysis
- High-performance numerical computations

Installation
-----------

.. code-block:: bash

   pip install cmb-analysis

Quick Start
----------

.. code-block:: python

   from cmb_analysis.cosmology import LCDM
   from cmb_analysis.analysis import PowerSpectrumCalculator
   from cmb_analysis.visualization import CMBPlotter

   # Initialize model
   model = LCDM()
   
   # Compute power spectra
   calculator = PowerSpectrumCalculator()
   spectra = calculator.compute_all_spectra(params)
   
   # Visualize results
   plotter = CMBPlotter()
   plotter.plot_power_spectra(spectra)

Contributing
-----------

We welcome contributions! Please see our :doc:`contributing` guide for details.

Citation
--------

If you use this software in your research, please cite:

.. code-block:: bibtex

   @software{cmb_analysis2024,
     author = {Your Name},
     title = {CMBAnalysis: A Modern Framework for CMB Analysis},
     year = {2024},
     url = {https://github.com/yourusername/cmb-analysis}
   }

Indices and Tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`