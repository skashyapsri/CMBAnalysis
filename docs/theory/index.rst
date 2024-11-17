Theoretical Background
====================

Comprehensive theoretical background for CMB analysis.

Cosmological Theory
-----------------

.. toctree::
   :maxdepth: 2

   cosmology/expansion
   cosmology/perturbations
   cosmology/recombination
   cosmology/inflation

CMB Physics
----------

.. toctree::
   :maxdepth: 2

   cmb/thermal_history
   cmb/acoustic_oscillations
   cmb/power_spectrum
   cmb/polarization

Statistical Methods
-----------------

.. toctree::
   :maxdepth: 2

   statistics/likelihood
   statistics/mcmc
   statistics/parameter_estimation
   statistics/error_analysis

Detailed Theory: CMB Power Spectrum
--------------------------------

The CMB power spectrum encodes crucial information about the early universe.

Temperature Power Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~

The temperature angular power spectrum :math:`C_\ell^{TT}` is defined as:

.. math::

   C_\ell^{TT} = \frac{2}{\pi} \int_0^\infty dk \, k^2 P(k) 
   \left|\Delta_\ell^T(k)\right|^2

where:

- :math:`P(k)` is the primordial power spectrum
- :math:`\Delta_\ell^T(k)` is the temperature transfer function
- :math:`\ell` is the multipole moment

Key Features
~~~~~~~~~~~

1. **Acoustic Peaks**: Reflect the size of the sound horizon at recombination
2. **Damping Tail**: Shows photon diffusion effects
3. **Large Scales**: Probe the Sachs-Wolfe plateau
4. **Reionization**: Affects low-ℓ polarization

Parameter Dependencies
~~~~~~~~~~~~~~~~~~~

The power spectrum shape depends on cosmological parameters:

- :math:`H_0`: Overall angular scaling
- :math:`\omega_b`: Relative peak heights
- :math:`\omega_{cdm}`: Peak locations
- :math:`\tau`: Reionization optical depth
- :math:`n_s`: Spectral tilt
- :math:`A_s`: Overall amplitude