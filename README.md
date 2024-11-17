# CMBAnalysis: Advanced Framework for Cosmic Microwave Background Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-astro--ph%2FXXXXXX-red.svg)](https://arxiv.org/abs/astro-ph/XXXXXX)

A comprehensive Python framework for analyzing Cosmic Microwave Background (CMB) radiation data, implementing modern MCMC techniques for cosmological parameter estimation and providing publication-quality visualization tools.

## Features

### Core Capabilities

- **Cosmological Models**
  - Standard ΛCDM implementation
  - Extended wCDM model
  - Support for custom model development
  - Robust numerical computations

### Analysis Tools

- **Power Spectrum Analysis**

  - Temperature (TT) correlations
  - E-mode polarization (EE)
  - Temperature-E-mode cross-correlations (TE)
  - Advanced transfer function calculations

- **Parameter Estimation**
  - MCMC implementation with emcee
  - Parallel tempering capabilities
  - Convergence diagnostics
  - Fisher matrix analysis

### Visualization

- **Publication-Quality Plots**
  - Power spectrum visualization
  - Parameter constraints
  - MCMC diagnostics
  - Custom plotting utilities

### Advanced Features

- Comprehensive error analysis
- Numerical stability safeguards
- High-performance computations
- Extensive documentation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cmb-analysis.git
cd cmb-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .[dev]
```

## Quick Start

```python
from cmb_analysis.cosmology import LCDM
from cmb_analysis.analysis import PowerSpectrumCalculator
from cmb_analysis.visualization import CMBPlotter

# Initialize model
model = LCDM()
params = {
    'H0': 67.32,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'tau': 0.0544,
    'ns': 0.9649,
    'ln10As': 3.044
}

# Compute power spectra
calculator = PowerSpectrumCalculator()
cl_tt, cl_ee, cl_te = calculator.compute_all_spectra(params)

# Create visualization
plotter = CMBPlotter()
plotter.plot_power_spectra({
    'cl_tt': cl_tt,
    'cl_ee': cl_ee,
    'cl_te': cl_te
})
```

## Project Structure

```
cmb_analysis/
├── cmb_analysis/
│   ├── cosmology/      # Cosmological models
│   ├── analysis/       # Analysis tools
│   ├── visualization/  # Plotting utilities
│   └── utils/          # Helper functions
├── docs/
│   ├── api/            # API documentation
│   ├── examples/       # Example code
│   ├── tutorials/      # Step-by-step guides
│   └── theory/         # Theoretical background
├── tests/              # Test suite
├── notebooks/          # Jupyter notebooks
└── examples/           # Example scripts
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)
- [Theory Background](docs/theory/)
- [Example Gallery](docs/examples/)

### Interactive Notebooks

1. [Quick Start Guide](notebooks/01_quick_start.ipynb)
2. [Parameter Estimation](notebooks/02_parameter_estimation.ipynb)
3. [Advanced Analysis](notebooks/03_advanced_analysis.ipynb)

## Development

### Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- emcee
- corner
- healpy

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=cmb_analysis tests/
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Academic Usage

If you use this code in your research, please cite:

```bibtex
@software{cmb_analysis2024,
  author = {Your Name},
  title = {CMBAnalysis: A Modern Framework for Cosmic Microwave Background Analysis},
  year = {2024},
  url = {https://github.com/yourusername/cmb-analysis},
  version = {1.0.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using [emcee](https://emcee.readthedocs.io/) for MCMC analysis
- Visualization powered by [matplotlib](https://matplotlib.org/) and [corner.py](https://corner.readthedocs.io/)
- Special thanks to the scientific computing community for their invaluable tools and libraries

## Contact

Your Name - your.email@institution.edu

Project Link: [https://github.com/yourusername/cmb-analysis](https://github.com/yourusername/cmb-analysis)
