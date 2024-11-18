# CMBAnalysis Project - Tech Doc

## I. Technical Framework

### A. Physics Implementation

1. **Transfer Functions**

   ```python
   def matter_transfer(k: ArrayLike, params: Dict[str, float]) -> ArrayLike:
       """
       Compute matter transfer function with neutrinos.
       T(k) = (1-f_ν)T_cb(k) + f_ν T_ν(k)
       """
       T_cb = self._compute_cdm_baryon_transfer(k, params)
       f_nu = self.m_nu_sum / (93.14 * params['omega_cdm'])
       T_nu = T_cb * self.neutrino_phase_space(k, params)
       return (1-f_nu)*T_cb + f_nu*T_nu
   ```

2. **Power Spectrum Computation**
   ```python
   def compute_all_spectra(self, params: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
       """
       Compute TT, EE, TE spectra:
       C_l^{XY} = 4π ∫ dk k² P(k) Δ_l^X(k) Δ_l^Y(k)
       """
       k = self.get_k_grid(self.ell_max)
       transfer = self.compute_transfer_functions(k, params)
       primordial = self.compute_primordial_spectrum(k, params)
       return self.integrate_spectra(k, transfer, primordial)
   ```

### B. Numerical Methods

1. **Integration Techniques**

   - Adaptive Gaussian quadrature for k-space
   - Bessel function optimization
   - Asymptotic approximations

   ```python
   def integrate_spectra(self, k: ArrayLike, transfer: Dict[str, ArrayLike],
                        P_k: ArrayLike) -> ArrayLike:
       """
       Integrate power spectra using adaptive methods:
       - Gaussian quadrature for well-behaved regions
       - Adaptive refinement for oscillatory parts
       - Richardson extrapolation for convergence
       """
   ```

2. **Stability Measures**
   - Regularization for small scales
   - Error propagation tracking
   - Numerical bounds checking

### C. MCMC Implementation

```python
class MCMCAnalysis:
    def __init__(self):
        self.sampler_config = {
            'nwalkers': 32,
            'nsteps': 2000,
            'moves': [
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2)
            ]
        }

    def setup_priors(self):
        """
        Set up parameter priors:
        - Gaussian for well-constrained parameters
        - Uniform for poorly constrained ones
        - Derived parameter handling
        """
```

## II. Data Handling Framework

### A. Planck Data Structure

```python
class PlanckDataHandler:
    """
    Handle Planck Legacy Archive data:
    - Power spectra (TT, TE, EE)
    - Covariance matrices
    - Beam and foreground corrections
    """
    def __init__(self, release: str = 'PR3'):
        self.data_paths = {
            'spectra': f'data/planck/{release}/power_spectra/raw',
            'cov': f'data/planck/{release}/covariance',
            'beam': f'data/planck/{release}/beam'
        }
```

### B. Cache System

```python
class ComputationCache:
    def __init__(self):
        self.cache_dir = Path('data/cache')
        self.transfer_cache = {}
        self.spectrum_cache = {}

    def get_cached_result(self, key: str, params: Dict[str, float]) -> Optional[ArrayLike]:
        """
        Retrieve cached computation with parameter hash checking
        """
```

## III. Implementation Details

### A. Cosmological Models

1. **ΛCDM Implementation**

   ```python
   class LCDM(CosmologyModel):
       def H(self, z: float) -> float:
           """
           Hubble parameter H(z) = H₀√[Ωm(1+z)³ + ΩΛ]
           """

       def angular_diameter_distance(self, z: float) -> float:
           """
           D_A(z) = c/H₀ ∫[dz'/E(z')] / (1+z)
           """
   ```

2. **wCDM Extensions**
   ```python
   class wCDM(LCDM):
       def H(self, z: float) -> float:
           """
           Extended Hubble parameter with w(z)
           H(z) = H₀√[Ωm(1+z)³ + ΩDE(1+z)^(3(1+w))]
           """
   ```

### B. Numerical Optimizations

1. **Transfer Function Computation**

   ```python
   def optimize_transfer_computation(self):
       """
       Optimization strategies:
       1. k-space gridding:
          - Linear for k < 0.1
          - Log-spaced for k > 0.1
       2. Bessel function approximations:
          - Small argument: j_l(x) ≈ x^l/(2l+1)!!
          - Large argument: j_l(x) ≈ sin(x-lπ/2)/x
       3. Cached intermediate results
       """
   ```

2. **Power Spectrum Integration**
   ```python
   def adaptive_k_integration(self):
       """
       Adaptive integration scheme:
       1. Identify oscillatory regions
       2. Adjust sampling density
       3. Use appropriate quadrature
       """
   ```

### C. Error Analysis

1. **Systematic Error Handling**

   ```python
   def propagate_errors(self):
       """
       Error propagation chain:
       1. Input parameter uncertainties
       2. Numerical integration errors
       3. Observational uncertainties
       4. Combined error estimation
       """
   ```

2. **Convergence Diagnostics**
   ```python
   def check_convergence(self):
       """
       Convergence criteria:
       1. Gelman-Rubin statistic < 1.1
       2. Effective sample size > 1000
       3. Parameter auto-correlation
       """
   ```

## IV. Implementation Notes

### A. Critical Considerations

1. **Numerical Stability**

   - Use log-space calculations where appropriate
   - Implement bounds checking at critical points
   - Monitor condition numbers in matrix operations

2. **Performance Optimization**

   - Cache frequently used computations
   - Vectorize operations where possible
   - Use parallel processing for independent calculations

3. **Accuracy Control**
   - Validate against known solutions
   - Compare with other codes (CLASS, CAMB)
   - Track error propagation

### B. Known Limitations

1. **Physical Limitations**

   - Maximum multipole ℓ ≤ 5000
   - Minimum k > 1e-5 Mpc⁻¹
   - Non-linear scales not fully modeled

2. **Computational Limitations**
   - Memory usage scales with ℓ_max
   - MCMC convergence time
   - Precision vs speed tradeoffs

### C. Future Improvements

1. **Short-term**

   - Optimize transfer function calculation
   - Implement parallel MCMC
   - Add more diagnostic tools

2. **Long-term**
   - Support for tensor modes
   - Extended cosmological models
   - Neural network emulator

## V. Technical Dependencies

### A. Core Requirements

```python
# requirements.txt
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
emcee>=3.1.0
corner>=2.2.0
healpy>=1.15.0
```

### B. Development Tools

```python
# dev-requirements.txt
pytest>=6.0
pytest-cov>=2.0
black>=22.0
mypy>=0.950
sphinx>=4.0
```

## VI. Testing Framework

### A. Test Categories

1. **Unit Tests**

   - Individual component functionality
   - Numerical accuracy
   - Edge cases

2. **Integration Tests**

   - End-to-end workflows
   - Data pipeline tests
   - Visualization tests

3. **Performance Tests**
   - Scaling tests
   - Memory usage
   - Computation time

### B. Test Data

```python
@pytest.fixture
def mock_cmb_data():
    """
    Generate mock CMB data with known properties
    for testing against analytical solutions
    """
```
