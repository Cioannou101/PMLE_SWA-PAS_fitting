# PMLE-SWA-PAS Fitting

A comprehensive Python toolkit for fitting Solar Wind particle velocity distribution functions (VDFs) using Poisson Maximum Likelihood Estimation (PMLE). This package analyzes data from the Solar Wind Analyser Proton and Alpha Sensor (SWA-PAS) instrument aboard the ESA Solar Orbiter spacecraft.

## Overview

This project implements advanced statistical methods to extract solar wind plasma parameters by fitting double bi-Maxwellian distribution models to measured velocity distribution function (VDF) data. The toolkit automatically identifies and separates core and beam populations in the proton distribution, accounting for drift characteristics and anisotropic temperatures.

**Key Features:**
- Poisson likelihood fitting framework
- Determination of core and beam proton bulk properties
- Three-step hierarchical fitting approach
- Goodness-of-fit metrics
- Integration with Solar Orbiter PAS/MAG data formats (CDF)
- Visualization tools for VDF comparison and fit validation

## Project Status

⚠️ **Work in Progress**: This codebase is actively maintained. Further documentation refinements and feature enhancements are ongoing.

📄 **Associated Publication**: A research paper documenting this fitting methodology has been submitted to *The Astrophysical Journal* (ApJ).

## Quick Start

### Prerequisites

- Python 3.7+
- NumPy, SciPy, Matplotlib
- lmfit (for optimization)
- cdflib (for CDF file I/O)
- h5py (for HDF5 output)
- pandas, scikit-learn (for data processing)

### Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install numpy scipy matplotlib lmfit cdflib h5py pandas scikit-learn tqdm cmocean
```

3. For data download capabilities, additionally install:
```bash
pip install sunpy sunpy-soar
```

### Basic Usage

#### 1. Learning the Fitting Method

Start with **`Fit_example_simulated.ipynb`** for a complete walkthrough:
- Demonstrates PMLE fitting on synthetic electrostatic analyzer VDF data
- Includes simulated data example with expected results
- Explains concept behind PMLE fitting

#### 2. Fitting Real PAS Data

Use **`fit_data_notebook.ipynb`** to:
- Load and preprocess Solar Orbiter PAS/MAG data
- Execute the full fitting pipeline
- Save results to HDF5 format for post-processing
- Generate comparison plots between fits and measurements

#### 3. Downloading Data

Data can be obtained via two methods:

**Option A: Manual Download**
1. Download CDF files from [Solar Orbiter SOAR portal](https://soar.esac.esa.int/)
2. Organize files in the following structure:
```
Data/
├── 2023_06_01/
│   ├── solo_L1_swa-pas-3d_20230601_*.cdf
│   ├── solo_L2_swa-pas-vdf_20230601_*.cdf
│   ├── solo_L2_swa-pas-grnd-mom_20230601_*.cdf
│   └── solo_L2_mag-srf-normal_20230601_*.cdf
└── 2023_06_02/
    └── ...
```

**Option B: Automated Download**
Edit and run `sunpy_soar_download.py`:
```python
# Modify date range and save path in the script
python sunpy_soar_download.py
```

## Project Structure

```
├── fit_data_notebook.ipynb          # Main analysis notebook for real data
├── Fit_example_simulated.ipynb      # Tutorial with simulated data
├── fit_models.py                    # Core PMLE fitting routines
├── Poisson_fit_functions.py         # Distribution model definitions
├── gen_funcs.py                     # Data preprocessing and utilities
├── load_data_functions.py           # CDF file processing for PAS/MAG data
├── gof_funcs.py                     # Goodness-of-fit metrics
├── plot_funcs.py                    # Visualization utilities
├── solo_spice.py                    # Spacecraft position/orientation tools
├── sunpy_soar_download.py           # Automated data download
├── Data/                            # Input CDF data directory
├── solo_spice/                      # SPICE kernels for orbit calculations
└── README.md                        # This file
```

## Typical Workflow

1. **Prepare Data**: Place CDF files in `Data/yyyy_mm_dd/` directories or use `sunpy_soar_download.py`
2. **Configure**: Edit `fit_data_notebook.ipynb` with your date range and parameters
3. **Run Fitting**: Execute the notebook to fit VDF data across all timestamps
4. **Inspect Results**: Review generated plots and HDF5 output files
5. **Post-Process**: Use output data for further analysis

## Output Format

Results are saved as HDF5 files containing:
- Fitted parameters (density, bulk velocity, temperatures, drift velocity)
- Goodness-of-fit metrics
- Timestamps and metadata

## Support & Contributing

For questions or issues using this code, please contact:
- **Author**: Charalambos Ioannou
- **Institution**: UCL / Mullard Space Science Laboratory
- **Email**: charalambos.ioannou.22@ucl.ac.uk
- **GitHub**: [@Cioannou101](https://github.com/Cioannou101)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

