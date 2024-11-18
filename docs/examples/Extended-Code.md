# CMBAnalysis - Extended Code Examples

## I. Core Physics Implementation

### A. Transfer Function Computation

```python
class TransferFunctionCalculator:
    def __init__(self):
        self.k_grid = np.logspace(-4, 2, 1000)  # Mpc^-1
        self.cache = {}

    def compute_transfer_function(self, k: np.ndarray, z: float,
                                params: Dict[str, float]) -> np.ndarray:
        """
        Compute full transfer function including all physical effects.

        Parameters
        ----------
        k : array_like
            Wavenumbers in h/Mpc
        z : float
            Redshift
        params : dict
            Cosmological parameters

        Returns
        -------
        array_like
            Transfer function T(k,z)
        """
        cache_key = self._get_cache_key(z, params)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute components
        T_cdm = self._compute_cdm_transfer(k, params)
        T_baryon = self._compute_baryon_transfer(k, params)
        T_nu = self._compute_neutrino_transfer(k, params)

        # Combine with proper weights
        omega_cdm = params['omega_cdm']
        omega_b = params['omega_b']
        omega_nu = self._get_neutrino_density(params)
        omega_m = omega_cdm + omega_b + omega_nu

        T = (omega_cdm * T_cdm + omega_b * T_baryon + omega_nu * T_nu) / omega_m

        # Apply growth factor
        D = self._compute_growth_factor(z, params)
        T *= D

        self.cache[cache_key] = T
        return T

    def _compute_cdm_transfer(self, k: np.ndarray,
                            params: Dict[str, float]) -> np.ndarray:
        """
        Compute CDM transfer function.

        Uses fitting formula from Eisenstein & Hu (1998).
        """
        q = k * (params['h']/0.02237)  # Scaled wavenumber

        # Compute characteristic scales
        k_eq = self._compute_equality_scale(params)
        alpha_c = (params['omega_cdm']/0.12)**0.67
        beta_c = 1 + (params['omega_b']/0.02237)**0.76

        # Transfer function
        T = np.log(2*np.e + 1.8*q)/(np.log(2*np.e + 1.8*q) + (14.2 + 386/(1+69.9*q**1.08))*q**2)

        return T

    def _compute_baryon_transfer(self, k: np.ndarray,
                               params: Dict[str, float]) -> np.ndarray:
        """
        Compute baryon transfer function including acoustic oscillations.
        """
        # Sound horizon
        rs = self._compute_sound_horizon(params)

        # Silk damping scale
        k_silk = self._compute_silk_scale(params)

        # Oscillatory part
        x = k * rs
        j0 = np.sin(x) / x  # Spherical Bessel function

        # Envelope
        silk_damping = np.exp(-(k/k_silk)**2)

        return j0 * silk_damping

    def _compute_growth_factor(self, z: float,
                             params: Dict[str, float]) -> float:
        """
        Compute linear growth factor D(z).
        """
        def growth_integrand(a: float) -> float:
            return (a * self._hubble(1/a - 1, params)/self._hubble(0, params))**-3

        a = 1/(1 + z)
        norm = 1/growth_integrand(1)  # Normalize to D(z=0) = 1

        integral = integrate.quad(growth_integrand, 0, a)[0]
        return a * integral * norm
```

### B. Power Spectrum Computation

```python
class PowerSpectrumCalculator:
    def __init__(self, transfer_calc: TransferFunctionCalculator):
        self.transfer_calc = transfer_calc

    def compute_matter_power(self, k: np.ndarray, z: float,
                           params: Dict[str, float]) -> np.ndarray:
        """
        Compute matter power spectrum P(k,z).

        P(k,z) = A_s * (k/k_pivot)^(n_s-1) * T(k,z)^2
        """
        # Primordial spectrum
        k_pivot = 0.05  # Mpc^-1
        A_s = np.exp(params['ln10As']) * 1e-10
        n_s = params['ns']
        P_prim = A_s * (k/k_pivot)**(n_s - 1)

        # Transfer function
        T = self.transfer_calc.compute_transfer_function(k, z, params)

        return P_prim * T**2

    def compute_cmb_spectra(self, params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Compute CMB temperature and polarization spectra.
        """
        ell = np.arange(2, 2500)
        k = np.logspace(-4, 2, 1000)

        # Get transfer functions
        Delta_T = self._compute_temperature_transfer(k, ell, params)
        Delta_E = self._compute_polarization_transfer(k, ell, params)

        # Primordial spectrum
        P_prim = self._compute_primordial_spectrum(k, params)

        # Compute spectra
        spectra = {}

        # TT spectrum
        spectra['cl_tt'] = self._integrate_spectrum(k, P_prim, Delta_T, Delta_T)

        # EE spectrum
        spectra['cl_ee'] = self._integrate_spectrum(k, P_prim, Delta_E, Delta_E)

        # TE cross-spectrum
        spectra['cl_te'] = self._integrate_spectrum(k, P_prim, Delta_T, Delta_E)

        return spectra

    def _integrate_spectrum(self, k: np.ndarray, P_k: np.ndarray,
                          Delta1: np.ndarray, Delta2: np.ndarray) -> np.ndarray:
        """
        Perform k-space integration for power spectra.
        """
        integrand = k**2 * P_k[:, np.newaxis] * Delta1 * Delta2

        # Use Simpson's rule for integration
        result = integrate.simps(integrand, np.log(k), axis=0)

        # Convert to C_ell
        return 4 * np.pi * result
```

### C. MCMC Implementation

```python
class MCMCAnalysis:
    def __init__(self, data: Dict[str, np.ndarray],
                 power_calc: PowerSpectrumCalculator):
        self.data = data
        self.power_calc = power_calc

        # MCMC settings
        self.nwalkers = 32
        self.nsteps = 2000

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """
        Compute log-likelihood for parameter set.
        """
        # Compute theory spectra
        theory = self.power_calc.compute_cmb_spectra(params)

        # Compute chi-square
        chi2 = 0
        for spec in ['tt', 'te', 'ee']:
            residuals = self.data[f'cl_{spec}'] - theory[f'cl_{spec}']
            chi2 += np.sum((residuals / self.data[f'error_{spec}'])**2)

        return -0.5 * chi2

    def log_prior(self, params: Dict[str, float]) -> float:
        """
        Compute log-prior for parameter set.
        """
        # Parameter bounds
        bounds = {
            'H0': (60, 80),
            'omega_b': (0.019, 0.025),
            'omega_cdm': (0.10, 0.14),
            'tau': (0.01, 0.10),
            'ns': (0.9, 1.0),
            'ln10As': (2.7, 3.3)
        }

        # Check bounds
        for param, (low, high) in bounds.items():
            if not low <= params[param] <= high:
                return -np.inf

        # Gaussian priors
        chi2_prior = 0
        priors = {
            'H0': (67.32, 0.54),
            'omega_b': (0.02237, 0.00015)
        }

        for param, (mean, std) in priors.items():
            chi2_prior += ((params[param] - mean) / std)**2

        return -0.5 * chi2_prior

    def log_probability(self, params: Dict[str, float]) -> float:
        """
        Compute total log-probability.
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(params)

    def run_mcmc(self) -> Dict[str, np.ndarray]:
        """
        Run MCMC analysis.
        """
        # Initialize walkers
        ndim = len(self.param_names)
        pos = self._initialize_walkers(ndim)

        # Set up sampler
        sampler = emcee.EnsembleSampler(
            self.nwalkers, ndim, self.log_probability
        )

        # Run MCMC
        sampler.run_mcmc(pos, self.nsteps, progress=True)

        return {
            'chain': sampler.get_chain(),
            'log_prob': sampler.get_log_prob()
        }
```

`````markdown
# CMBAnalysis - Theoretical Background

## I. Cosmological Framework

### A. Background Evolution

The universe's expansion is described by the Friedmann equations:

1. **First Friedmann Equation**

   ```python
   def H2(z: float, params: Dict[str, float]) -> float:
       """
       Compute H²/H₀² = Ωm(1+z)³ + Ωr(1+z)⁴ + ΩΛ
       """
       omega_m = params['omega_m']
       omega_r = params['omega_r']
       omega_lambda = 1 - omega_m - omega_r

       return (omega_m * (1+z)**3 +
               omega_r * (1+z)**4 +
               omega_lambda)
   ```

2. **Scale Factor Evolution**

   ```python
   def scale_factor_evolution(t: float, params: Dict[str, float]) -> float:
       """
       Solve ä/a = -4πG/3(ρ + 3p)
       """
       def deriv(y: np.ndarray, t: float) -> np.ndarray:
           a, adot = y
           H2 = self.H2(1/a - 1, params)
           return [adot, -0.5 * a * H2]

       return integrate.solve_ivp(deriv, (0, t), y0=[1e-6, 1e-6])
   ```

### B. Perturbation Theory

1. **Gauge-Invariant Formalism**

   ```python
   class PerturbationEvolution:
       def __init__(self):
           self.k_grid = np.logspace(-4, 2, 1000)

       def compute_gauge_invariant_variables(self,
           phi: np.ndarray,
           delta: np.ndarray,
           theta: np.ndarray) -> Dict[str, np.ndarray]:
           """
           Compute gauge-invariant perturbations:
           Ψ = φ + H(v_B - B)
           Φ = η - H(v_B - B)
           """
           return {
               'Psi': self._compute_psi(phi, theta),
               'Phi': self._compute_phi(phi, delta)
           }
   ```

2. **Einstein-Boltzmann System**

   ```python
   class EinsteinBoltzmannSolver:
       def evolve_perturbations(self, k: float, z: float) -> Dict[str, np.ndarray]:
           """
           Solve coupled Einstein-Boltzmann equations:

           Photons (Θ_l):
           Θ̇₀ = -kΘ₁ - Φ̇
           Θ̇₁ = k(Θ₀ + Ψ - Θ₂/2)
           Θ̇ₗ = k[lΘ_{l-1} - (l+1)Θ_{l+1}]/(2l+1)

           Baryons:
           δ̇ᵦ = -θᵦ + 3Φ̇
           θ̇ᵦ = -Hθᵦ + kΨ + R⁻¹κ̇(θᵧ - θᵦ)
           """
           # Set up initial conditions
           y0 = self._initialize_perturbations()

           # Define system of equations
           def derivs(tau: float, y: np.ndarray) -> np.ndarray:
               return self._compute_derivatives(tau, y, k)

           # Solve system
           sol = integrate.solve_ivp(
               derivs,
               t_span=(0, self.tau_final),
               y0=y0,
               method='LSODA',
               rtol=1e-8
           )

           return self._process_solution(sol)
   ```

### C. CMB Physics

1. **Temperature Anisotropies**

   ```python
   class CMBAnisotropies:
       def compute_source_function(self, k: float, tau: float) -> Dict[str, float]:
           """
           Compute source functions for temperature and polarization:

           ST = g(Θ₀ + Ψ) + e⁻ᵗ(Ψ̇ + Φ̇) + gv_b
           SE = g(Θ₂ + P₂)
           """
           # Visibility function
           g = self.visibility_function(tau)
           gdot = self.visibility_function_derivative(tau)

           # Metric perturbations
           psi = self.metric_perturbations['psi'](k, tau)
           phi = self.metric_perturbations['phi'](k, tau)

           # Source terms
           ST = (g * (self.Theta0(k, tau) + psi) +
                np.exp(-self.optical_depth(tau)) *
                (self.psi_dot(k, tau) + self.phi_dot(k, tau)) +
                g * self.baryon_velocity(k, tau))

           SE = g * (self.Theta2(k, tau) + self.P2(k, tau))

           return {'temperature': ST, 'polarization': SE}
   ```

2. **Line of Sight Integration**

   ```python
   class LineOfSightIntegrator:
       def compute_transfer_function(self, k: float, l: int) -> Dict[str, float]:
           """
           Compute transfer functions using line of sight integration:

           Δₗ(k) = ∫dτ S(k,τ) jₗ(k(τ₀-τ))
           """
           def integrand(tau: float) -> float:
               # Source function
               S = self.source_function(k, tau)

               # Spherical Bessel function
               x = k * (self.tau0 - tau)
               j_l = self._spherical_bessel(l, x)

               return S * j_l

           result = integrate.quad(integrand, 0, self.tau0,
                                 epsabs=1e-8, epsrel=1e-8)
           return result[0]
   ```

### D. Statistical Analysis

1. **Power Spectrum Estimation**

   ```python
   class PowerSpectrumEstimator:
       def compute_angular_power_spectrum(self,
           alm: np.ndarray,
           lmax: int) -> np.ndarray:
           """
           Compute Cₗ from aₗₘ coefficients:
           Cₗ = 1/(2l+1) Σₘ |aₗₘ|²
           """
           cls = np.zeros(lmax + 1)
           for l in range(2, lmax + 1):
               m_vals = np.arange(-l, l+1)
               cls[l] = np.mean(np.abs(alm[l][m_vals + l])**2)
           return cls
   ```

2. **Likelihood Analysis**

   ```python
   class CMBLikelihood:
       def __init__(self, data: Dict[str, np.ndarray],
                   covariance: np.ndarray):
           self.data = data
           self.covariance = covariance
           self.inv_cov = np.linalg.inv(covariance)

       def compute_likelihood(self, theory: Dict[str, np.ndarray]) -> float:
           """
           Compute likelihood:
           -2ln L = (x-μ)ᵀC⁻¹(x-μ) + ln|C|
           """
           # Compute residuals
           residuals = np.concatenate([
               self.data[key] - theory[key]
               for key in ['tt', 'te', 'ee']
           ])

           # Compute chi-square
           chi2 = residuals @ self.inv_cov @ residuals

           # Add determinant term
           log_det = np.linalg.slogdet(self.covariance)[1]

           return -0.5 * (chi2 + log_det)
   ```

## II. Numerical Methods

### A. Integration Techniques

```python
class NumericalIntegration:
    def adaptive_integration(self,
        func: Callable,
        a: float,
        b: float,
        tol: float = 1e-8) -> float:
        """
        Adaptive integration with error control:
        - Gauss-Kronrod quadrature for smooth regions
        - Adaptive subdivision for oscillatory regions
        """
        def error_estimate(h: float, f: np.ndarray) -> float:
            """Estimate truncation error"""
            return h * np.abs(np.diff(f)).max()

        def should_subdivide(error: float) -> bool:
            """Decide if interval needs subdivision"""
            return error > tol

        return self._adaptive_quad(func, a, b, tol)
```

````markdown
## III. Advanced Physical Effects

### A. Reionization

```python
class ReionizationModel:
    def __init__(self):
        self.z_reio_mean = 7.67  # Mean reionization redshift
        self.delta_z = 0.5       # Width of reionization

    def ionization_fraction(self, z: np.ndarray) -> np.ndarray:
        """
        Compute ionization fraction x_e(z) using tanh model:
        x_e(z) = 0.5[1 + tanh((z_reio - z)/Δz)]
        """
        return 0.5 * (1 + np.tanh((self.z_reio_mean - z) / self.delta_z))

    def optical_depth(self, z: np.ndarray) -> np.ndarray:
        """
        Compute optical depth τ(z):
        τ(z) = σᵧnₑ∫dx x_e(x)/H(x)(1+x)²
        """
        def integrand(x: float) -> float:
            return (self.ionization_fraction(x) *
                   self._electron_density(x) /
                   self.hubble(x) * (1 + x)**2)

        tau = np.zeros_like(z)
        for i, zi in enumerate(z):
            tau[i] = self.sigma_t * integrate.quad(
                integrand, 0, zi,
                epsrel=1e-8
            )[0]

        return tau
```
````
`````

### B. Neutrino Physics

```python
class NeutrinoEffects:
    def __init__(self):
        self.N_eff = 3.046      # Effective number of neutrinos
        self.m_nu_sum = 0.06    # Sum of neutrino masses (eV)

    def phase_space_integral(self, q: np.ndarray) -> np.ndarray:
        """
        Compute neutrino phase space integral:
        I(q) = ∫dp p²f(p)/E(p)
        """
        def integrand(p: float, q: float) -> float:
            E = np.sqrt(p**2 + self.m_nu_sum**2/9)  # Single neutrino mass
            f = 1 / (np.exp(p) + 1)  # Fermi-Dirac distribution
            return p**2 * f / E

        result = np.zeros_like(q)
        for i, qi in enumerate(q):
            result[i] = integrate.quad(
                integrand, 0, np.inf,
                args=(qi,),
                epsrel=1e-8
            )[0]

        return result

    def neutrino_velocity(self, k: np.ndarray, z: float) -> np.ndarray:
        """
        Compute neutrino velocity dispersion:
        v²(k,z) = k²/(1+z)²∫dq q²f(q)/∫dq q²f(q)E(q)
        """
        q = np.linspace(0, 10, 100)
        f = 1 / (np.exp(q) + 1)
        E = np.sqrt(q**2 + (self.m_nu_sum/self.T_nu)**2)

        numerator = np.trapz(q**2 * f, q)
        denominator = np.trapz(q**2 * f * E, q)

        return k**2 / (1+z)**2 * numerator/denominator
```

### C. Gravitational Lensing

```python
class LensingPotential:
    def compute_lensing_potential(self,
                                z_source: float,
                                params: Dict[str, float]) -> np.ndarray:
        """
        Compute lensing potential power spectrum:
        C_ℓ^ΦΦ = 16π∫dz[W(z)]²P_m(k=ℓ/χ,z)/χ²

        where W(z) is the lensing kernel:
        W(z) = χ(z*-z)/χ(z*) χ(z)
        """
        # Set up z integration
        z_arr = np.linspace(0, z_source, 1000)
        chi = self.comoving_distance(z_arr)

        # Compute lensing kernel
        W = self._lensing_kernel(z_arr, z_source)

        # Matter power spectrum at each z
        P_m = np.array([
            self.matter_power(k=self.ell/chi[i], z=z, params=params)
            for i, z in enumerate(z_arr)
        ])

        # Integrate
        integrand = W**2 * P_m / chi**2
        C_ell_phi = 16 * np.pi * np.trapz(integrand, z_arr)

        return C_ell_phi

    def _lensing_kernel(self, z: np.ndarray,
                       z_source: float) -> np.ndarray:
        """Compute lensing kernel W(z)"""
        chi = self.comoving_distance(z)
        chi_source = self.comoving_distance(z_source)

        return (chi_source - chi)/chi_source * chi
```

### D. Non-Linear Effects

```python
class NonLinearCorrections:
    def halofit_correction(self, k: np.ndarray, z: float,
                          P_linear: np.ndarray) -> np.ndarray:
        """
        Apply HALOFIT corrections to linear power spectrum
        following Smith et al. (2003)
        """
        # Compute nonlinear scale
        k_nl = self._find_nonlinear_scale(k, P_linear)

        # Get HALOFIT parameters
        params = self._get_halofit_params(k, P_linear, k_nl)

        # Compute quasi-linear term
        P_q = self._quasi_linear_power(k, P_linear, params)

        # Compute halo term
        P_h = self._halo_power(k, params)

        # Combine terms
        return P_q + P_h

    def _find_nonlinear_scale(self, k: np.ndarray,
                             P: np.ndarray) -> float:
        """
        Find scale where σ(R)=1:
        σ²(R) = 1/(2π²)∫dk k²P(k)W²(kR)
        """
        def sigma_squared(R: float) -> float:
            W = 3*(np.sin(k*R) - k*R*np.cos(k*R))/(k*R)**3
            integrand = k**2 * P * W**2 / (2*np.pi**2)
            return np.trapz(integrand, k)

        # Find root of σ(R) = 1
        from scipy.optimize import brentq
        R_nl = brentq(lambda R: sigma_squared(R) - 1, 0.01, 100)

        return 1/R_nl
```

## IV. Error Analysis and Uncertainty Propagation

### A. Systematic Effects

```python
class SystematicErrors:
    def beam_uncertainty(self, ell: np.ndarray,
                        fwhm: float,
                        dfwhm: float) -> np.ndarray:
        """
        Compute beam uncertainty:
        ΔC_ℓ/C_ℓ = 2ℓ(ℓ+1)σ_b²
        """
        sigma_b = fwhm / np.sqrt(8 * np.log(2))
        dsigma_b = dfwhm / np.sqrt(8 * np.log(2))

        return 2 * ell * (ell + 1) * sigma_b * dsigma_b

    def calibration_uncertainty(self,
                              C_ell: np.ndarray,
                              dcal: float) -> np.ndarray:
        """
        Compute calibration uncertainty:
        ΔC_ℓ = 2C_ℓ δg/g
        """
        return 2 * C_ell * dcal
```

### B. Covariance Matrix Estimation

```python
class CovarianceEstimation:
    def compute_covariance(self, C_ell: Dict[str, np.ndarray],
                          f_sky: float) -> np.ndarray:
        """
        Compute power spectrum covariance matrix:
        Cov(C_ℓ^{XY}, C_ℓ^{X'Y'}) =
            [(C_ℓ^{XX'}C_ℓ^{YY'} + C_ℓ^{XY'}C_ℓ^{X'Y})] /
            [(2ℓ+1)f_sky]
        """
        # Get all spectra
        C_tt = C_ell['tt']
        C_te = C_ell['te']
        C_ee = C_ell['ee']

        # Initialize covariance matrix
        n_ell = len(C_tt)
        cov = np.zeros((3*n_ell, 3*n_ell))

        # Fill covariance blocks
        for l in range(n_ell):
            # Factor from sky coverage and modes
            factor = 2/(2*l + 1)/f_sky

            # TT-TT block
            cov[l, l] = 2 * C_tt[l]**2 * factor

            # TE-TE block
            cov[n_ell+l, n_ell+l] = (
                (C_tt[l]*C_ee[l] + C_te[l]**2) * factor
            )

            # EE-EE block
            cov[2*n_ell+l, 2*n_ell+l] = 2 * C_ee[l]**2 * factor

            # Cross-correlations...

        return cov
```
