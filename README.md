# PMLE_SWA-PAS_fitting
Tools for performing Poisson Maximum Likelihood Estimation (PMLE) fitting on Solar Wind Analyser (SWA) Proton and Alpha Sensor (PAS) data from Solar Orbiter. This notebook fits a double bi-Maxwellian model on PAS Count and VDF data, yielding the bulk parameters of core and beam protons.

# Getting started

The Jupyter notebook Fit_example_simulated.ipynb explains how to apply PMLE fitting on electrostatic analyser VDF data, and shows an example on simulated data.

# Fitting PAS data

## Download data

Download your data in a "Data" folder in this format: "Data/yyyy_mm_dd/"
The data files required are: -swa-pas-3d, -swa-pas-vdf, -swa-pas-grnd-mom, -mag_srf

You can also run the sunpy_soar_download.py, adjusting the time interval and savepath as needed.

## Perform the fit

The notebook fit_data_notebook.ipynb will load the data, perform the necessary preprocessing and apply the fitting routine, saving the resuls using h5py. The notebook also can load the data and plot the fits against the data at specified timestamps.

# Status

⚠️ This project is currently a work in progress and will be further cleaned, documented, and polished in future updates.

If you are interested in using this code, contact charalambos.ioannou.22@ucl.ac.uk.

A research paper using this code has been submitted to The Astrophysical Journal (ApJ).
