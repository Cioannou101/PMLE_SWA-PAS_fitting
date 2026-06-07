"""
Author: Charalambos Ioannou
Institution: UCL / MSSL
Email: charalambos.ioannou.22@ucl.ac.uk
GitHub: @Cioannou101
Created: 2026-06-07

This script contains functions for fitting the VDF data with Poisson likelihood fits, including various VDF models (core only, core+beam with parallel drift), and a three step fitting approach for the core+beam fit.
"""

import os
import numpy as np
import lmfit
from tqdm import tqdm
import scipy.constants as sc
from Poisson_fit_functions import *
from gen_funcs import *
from gof_funcs import *
from datetime import datetime
import h5py

def fit_one_core_sep_par_drift_h5(tasks):
    """
    Fitting function for separate core and beam fit, only parallel drift. 
    Beam direction is picked automatically based on the moments bulk velocity and the velocity where the distribution peaks!
    Three step approach. First only core fit on select data. Second, fit beam with set core. Third, fit core again with set beam.
    """
    
    ind, t_vdf, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels, R_dist, qf = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    if np.all((counts_in == 0) | np.isnan(counts_in)):
        return ind, t_vdf, np.full(12, np.nan), param_names, np.full(12, np.nan), np.full(12, np.nan), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, R_dist, n, v_bulk_bf, T_par, T_perp, B_all, qf

    try:
        uxt, uyt, uzt = vx_bf, vy_bf, vz_bf
        umag = np.linalg.norm([uxt, uyt, uzt], axis = 0)
        # upar = uxt - v_bulk_bf[0] 
        # uperp = np.linalg.norm([uyt - v_bulk_bf[1], uzt - v_bulk_bf[2]])
        
        counts_core = np.copy(counts_in)
        # core_mask = upar < 0

        data_f = np.copy(vdf_in)
        data_integrated = integrate_vdf_over_angles(data_f, theta)
        ind_max = np.nanargmax(data_integrated)

        if np.linalg.norm(v_bulk_bf) < vels[ind_max] - 25e3:
            core_mask = umag > np.linalg.norm(v_bulk_bf)
            vxb_init = -1.0

        elif np.linalg.norm(v_bulk_bf) > vels[ind_max] - 25e3:    
            core_mask = umag < np.linalg.norm(v_bulk_bf)
            vxb_init = 1.0

        counts_core[~core_mask] = np.nan
    
    # if core data proccessing fails, return NaNs
    except Exception as e:
        print(f"Error processing counts_core at index {ind}: {e}")
        return ind, t_vdf, np.full(12, np.nan), param_names, np.full(12, np.nan), np.full(12, np.nan), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, R_dist, n, v_bulk_bf, T_par, T_perp, B_all, qf
        
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_core, G_factors)
    
    # Velocities
    ux, uy, uz = vx_bf, vy_bf, vz_bf
    
    n_in = n * 1e6
    T_par_in, T_perp_in, T_in = T_par, T_perp, T
    vth_par_in = np.sqrt(2 * T_par_in * sc.e / sc.m_p)
    vth_perp_in = np.sqrt(2 * T_perp_in * sc.e / sc.m_p)
    vth_in = np.sqrt(2 * T_in * sc.e / sc.m_p)
    vx_in, vy_in, vz_in = v_bulk_bf
    B_in = B_all * 1e-9
    VA_in = np.linalg.norm(B_in) / np.sqrt(sc.mu_0 * sc.m_p * n_in)
    
    # Initial fit parameters
    n_core = 0.8 * n_in
    vx_core = vx_in * 0.95
    vth_par_core = vth_perp_in * 0.85
    vth_perp_core = vth_perp_in
    
    constraints_min = np.zeros(12)
    constraints_max = np.zeros(12)
    
    # Core Constraints
    constraints_min[0] = 0.5 * n_in
    constraints_min[1] = vx_core - VA_in
    constraints_min[2] = vy_in - 50e3
    constraints_min[3] = vz_in - 50e3
    constraints_min[4] = 0.5 * vth_par_core
    constraints_min[5] = 0.5 * vth_perp_core
    
    constraints_max[0] = 1.1 * n_in
    constraints_max[1] = vx_core + VA_in
    constraints_max[2] = vy_in + 50e3
    constraints_max[3] = vz_in + 50e3
    constraints_max[4] = 2 * vth_par_core
    constraints_max[5] = 2 * vth_perp_core
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_core, min=constraints_min[0], max=nc_init * n_in)
    fit_params.add('vx', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    success = np.zeros(3)
    success[0] = minimiser_core.success
    # results_core.append(minimiser_core)
    
    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])
    
    (
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    ) = opt_params_core

    "Now fit beam with set core"
    
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_in, G_factors)
    
    # Initial fit parameters
    
    n_beam = 0.2 * n_in
    vx_beam = vx_fit_c + (vxb_init * np.sign(vx_fit_c) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    # Beam Constraints
    constraints_min[6] = 0
    constraints_min[7] = vx_beam - VA_in
    constraints_min[8] = vy_in - 50e3
    constraints_min[9] = vz_in - 50e3
    constraints_min[10] = 0.5 * vth_par_beam
    constraints_min[11] = 0.5 * vth_perp_beam
    
    constraints_max[6] = 0.5 * n_in
    constraints_max[7] = vx_beam + VA_in
    constraints_max[8] = vy_in + 50e3
    constraints_max[9] = vz_in + 50e3
    constraints_max[10] = 2 * vth_par_beam
    constraints_max[11] = 2 * vth_perp_beam

    # Initial Conditions
    in_conds = np.zeros(12)
    in_conds[0] = n_core
    in_conds[1] = vx_core
    in_conds[2] = vy_in
    in_conds[3] = vz_in
    in_conds[4] = vth_par_core
    in_conds[5] = vth_perp_core
    in_conds[6] = n_beam
    in_conds[7] = vx_beam
    in_conds[8] = vy_in
    in_conds[9] = vz_in
    in_conds[10] = vth_par_beam
    in_conds[11] = vth_perp_beam
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_fit_c, vary = False)
    fit_params.add('vx1', value=vx_fit_c, vary = False)
    fit_params.add('vy', value=vy_fit_c, vary = False)
    fit_params.add('vz', value=vz_fit_c,  vary = False)
    fit_params.add('v_th_par1', value=vth_par_fit_c, vary = False)
    fit_params.add('v_th_perp1', value=vth_perp_fit_c, vary = False)
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])
    
    # Minimise
    minimiser_beam = lmfit.minimize(
        logp_double_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    success[1] = minimiser_beam.success
    
    # Extract optimised parameters
    opt_params_beam = np.array([minimiser_beam.params[name].value for name in minimiser_beam.var_names])
    
    (
    n_fit_b, vx_fit_b, vth_par_fit_b, vth_perp_fit_b
    ) = opt_params_beam
    
    "Now fit core again with set beam"
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_fit_c, min=constraints_min[0], max=constraints_max[0])
    fit_params.add('vx1', value=vx_fit_c, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_fit_c, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_fit_c, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par1', value=vth_par_fit_c, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp1', value=vth_perp_fit_c, min=constraints_min[5], max=constraints_max[5])
    
    fit_params.add('n_var2', value=n_fit_b, vary = False)
    fit_params.add('vx2', value=vx_fit_b, vary = False)
    fit_params.add('v_th_par2', value=vth_par_fit_b, vary = False)
    fit_params.add('v_th_perp2', value=vth_perp_fit_b, vary = False)
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_double_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    opt_params = np.zeros(len(minimiser_core.params))
    for i, j in enumerate(minimiser_core.params):
        opt_params[i] = minimiser_core.params[j].value
    
    success[2] = minimiser_core.success

    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    opt_params_new = np.array([n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b])

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    "Compute goodness of fit"
    ks_val, p_val = ks_gof(counts_data, (ux, uy, uz, G_in, dt), opt_params, seed=123)

    D_norm, df, L_mean, L_std = dev_gof(counts_data, (ux, uy, uz, G_in, dt), opt_params)

    counts_save = np.array([np.nanmax(counts_data), np.nansum(counts_data), df + 10])

    param_names = ['n_c', 'vx_c', 'vy_c', 'vz_c', 'vth_par_c', 'vth_perp_c', 'n_b', 'vx_b', 'vy_b', 'vz_b', 'vth_par_b', 'vth_perp_b']

    return (ind, t_vdf, opt_params_new, param_names, in_conds, constraint_flag_all, overlap_3d, ks_val, p_val, D_norm, L_mean, L_std, R_dist, n, v_bulk_bf, T_par, T_perp, B_all, qf, success, counts_save)

def fit_one_core_only_h5(tasks):
    """
    Fitting function for core only Bi-Maxwellian fit.

    """
    
    ind, t_vdf, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels, R_dist, qf = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    if np.all((counts_in == 0) | np.isnan(counts_in)):
        return ind, t_vdf, np.full(6, np.nan), param_names, np.full(6, np.nan), np.full(6, np.nan), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, R_dist, n, v_bulk_bf, T_par, T_perp, B_all, qf
        
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_in, G_factors)
    
    # Velocities
    ux, uy, uz = vx_bf, vy_bf, vz_bf
    
    n_in = n * 1e6
    T_par_in, T_perp_in, T_in = T_par, T_perp, T
    vth_par_in = np.sqrt(2 * T_par_in * sc.e / sc.m_p)
    vth_perp_in = np.sqrt(2 * T_perp_in * sc.e / sc.m_p)
    vth_in = np.sqrt(2 * T_in * sc.e / sc.m_p)
    vx_in, vy_in, vz_in = v_bulk_bf
    B_in = B_all * 1e-9
    VA_in = np.linalg.norm(B_in) / np.sqrt(sc.mu_0 * sc.m_p * n_in)
    
    # Initial fit parameters
    n_core = 0.95 * n_in
    vx_core = vx_in * 1
    vth_par_core = vth_par_in * 0.9
    vth_perp_core = vth_perp_in
    
    constraints_min = np.zeros(6)
    constraints_max = np.zeros(6)
    
    # Core Constraints
    constraints_min[0] = 0.5 * n_in
    constraints_min[1] = vx_core - VA_in
    constraints_min[2] = vy_in - 50e3
    constraints_min[3] = vz_in - 50e3
    constraints_min[4] = 0.5 * vth_par_core
    constraints_min[5] = 0.5 * vth_perp_core
    
    constraints_max[0] = 1.5 * n_in
    constraints_max[1] = vx_core + VA_in
    constraints_max[2] = vy_in + 50e3
    constraints_max[3] = vz_in + 50e3
    constraints_max[4] = 2 * vth_par_core
    constraints_max[5] = 2 * vth_perp_core

    # Initial Conditions
    in_conds = np.zeros(6)
    in_conds[0] = n_core
    in_conds[1] = vx_core
    in_conds[2] = vy_in
    in_conds[3] = vz_in
    in_conds[4] = vth_par_core
    in_conds[5] = vth_perp_core

    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_core, min=constraints_min[0], max=nc_init * n_in)
    fit_params.add('vx', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    # results_core.append(minimiser_core)
    # success flag for core only fit replicated to match shape of double fit success flag
    success = np.array([minimiser_core.success, minimiser_core.success, minimiser_core.success])

    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_core) | (constraints_min == opt_params_core)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"

    overlap_3d = np.nan

    "Compute goodness of fit"
    ks_val, p_val = ks_gof_core_only(counts_data, (ux, uy, uz, G_in, dt), opt_params_core, seed=123)

    D_norm, df, L_mean, L_std = dev_gof_core_only(counts_data, (ux, uy, uz, G_in, dt), opt_params_core)

    counts_save = np.array([np.nanmax(counts_data), np.nansum(counts_data), df + 10])

    param_names = ['n_c', 'vx_c', 'vy_c', 'vz_c', 'vth_par_c', 'vth_perp_c']

    return (ind, t_vdf, opt_params_core, param_names, in_conds, constraint_flag_all, overlap_3d, ks_val, p_val, D_norm, L_mean, L_std, R_dist, n, v_bulk_bf, T_par, T_perp, B_all, qf, success, counts_save)

def save_all_results_h5(
    h5_path,
    times,          # (N, 1) float64 (timestamps)
    opt_params,     # (N, P) float64
    param_names,    # list of P strings
    in_conds,       # (N, P) float64
    constraints,    # (N, P) float64
    overlap,        # (N, 1) float64
    ks_metric,      # (N, 2) float64
    D_metric,       # (N, 3) float
    R,              # (N, 1) float64
    n_mom,          # (N, 1) float64
    v_mom,          # (N, 3) float64
    T_mom,          # (N, 2) float64
    B,              # (N, 3) float64
    qf,             # (N, 1) float64
    success,        # (N, 3) bool
    counts_save     # (N, 3) float64
):
    """
    Append all results to an existing HDF5 file (or create if not exists).
    Only saves unique time instances and associated data.
    """

    times = np.asarray(times, dtype=np.float64).reshape(-1, 1)
    opt_params = np.asarray(opt_params, dtype=np.float64)
    in_conds = np.asarray(in_conds, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    ks_metric = np.asarray(ks_metric, dtype=np.float64)
    D_metric = np.asarray(D_metric, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_mom = np.asarray(n_mom, dtype=np.float64)
    v_mom = np.asarray(v_mom, dtype=np.float64)
    T_mom = np.asarray(T_mom, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    qf = np.asarray(qf, dtype=np.float64)
    success = np.asarray(success, dtype=np.bool)
    counts_save = np.asarray(counts_save, dtype=np.float64)

    # --- Filter for unique times ---
    _, unique_indices = np.unique(times, return_index=True)
    times = times[unique_indices]
    opt_params = opt_params[unique_indices]
    in_conds = in_conds[unique_indices]
    constraints = constraints[unique_indices]
    overlap = overlap[unique_indices]
    ks_metric = ks_metric[unique_indices]
    D_metric = D_metric[unique_indices]
    R = R[unique_indices]
    n_mom = n_mom[unique_indices]
    v_mom = v_mom[unique_indices]
    T_mom = T_mom[unique_indices]
    B = B[unique_indices]
    qf = qf[unique_indices]
    success = success[unique_indices]
    counts_save = counts_save[unique_indices]

    # print("opt_params shape:", opt_params.shape)
    # print("in_conds shape:", in_conds.shape)
    # print("constraints shape:", constraints.shape)
    # print("overlap shape:", overlap.shape)
    # print("ks_metric shape:", ks_metric.shape)
    # print("R shape:", R.shape)
    # print("n_mom shape:", n_mom.shape)
    # print("v_mom shape:", v_mom.shape)
    # print("T_mom shape:", T_mom.shape)
    # print("B shape:", B.shape)

    mode = "a" if os.path.exists(h5_path) else "w"
    with h5py.File(h5_path, mode) as f:

        def append_ds(name, data, dedup=False):
            """Append or create dataset in file."""
            if name not in f:
                maxshape = (None,) + data.shape[1:]
                return f.create_dataset(name, data=data,
                                        maxshape=maxshape,
                                        chunks=True,
                                        compression="gzip")
            else:
                ds = f[name]
                new_data = data

                if dedup:  # only for "time"
                    existing = ds[:]
                    mask_new = ~np.isin(new_data[:, 0], existing[:, 0])
                    new_data = new_data[mask_new]

                if new_data.shape[0] > 0:
                    old = ds.shape[0]
                    ds.resize(old + new_data.shape[0], axis=0)
                    ds[old:] = new_data

                return ds

        # Write datasets
        dset_params = append_ds("fit_parameters", opt_params)
        if "param_names" not in dset_params.attrs:
            dset_params.attrs["param_names"] = param_names

        append_ds("initial_conditions", in_conds)
        append_ds("constraints", constraints)
        append_ds("overlap", overlap)
        append_ds("ks_metric", ks_metric)
        append_ds("D_metric", D_metric)
        append_ds("time", times)
        append_ds("R", R)
        append_ds('n_mom', n_mom)
        append_ds('v_mom', v_mom)
        append_ds('T_mom', T_mom)
        append_ds('B', B)
        append_ds('qf', qf)
        append_ds('success', success)
        append_ds('counts_save', counts_save)

def run_serial_h5(tasks, fun, h5_path, core_only=False):
    """
    Run tasks serially, collect results, and save them into HDF5
    at the end using `save_all_results_h5`..
    """

    results = []

    for ind_fit, task in enumerate(tqdm(tasks, desc="Fitting progress")):
        try:
            res = fun(task)
            results.append(res)

        except Exception as exc:
            print("Task error:", exc)
            # Create dummy result with NaNs
            t_nan = task[1]  # fallback timestamp
            R_nan = task[-1]  # fallback R_dist
            if core_only:
                n_params = 6
                opt_params_nan = np.full((n_params,), np.nan)
                in_conds_nan = np.full((n_params,), np.nan)
                constraints_nan = np.full((n_params,), np.nan)
                param_names_nan = ['n_c', 'vx_c', 'vy_c', 'vz_c', 'vth_par_c', 'vth_perp_c']
            else:
                n_params = 12 
                opt_params_nan = np.full((n_params,), np.nan)
                in_conds_nan = np.full((n_params,), np.nan)
                constraints_nan = np.full((n_params,), np.nan)
                param_names_nan = [
                    'n_c', 'vx_c', 'vy_c', 'vz_c', 'vth_par_c', 'vth_perp_c',
                    'n_b', 'vx_b', 'vy_b', 'vz_b', 'vth_par_b', 'vth_perp_b'
                ]
            v_nan = np.full((3), np.nan)
            B_nan = np.full((3), np.nan)
            success_nan = np.full((3), True)
            counts_nan = np.full((3), np.nan)

            dummy_res = (
                None,                 # ind (unused in save)
                t_nan,                # timestamp
                opt_params_nan,       # opt_params
                param_names_nan,
                in_conds_nan,         # in_conds
                constraints_nan,      # constraint_flag
                np.nan,               # overlap_3d
                np.nan,               # ks_val
                np.nan,               # p_val
                np.nan,               # D_norm
                np.nan,               # L_mean
                np.nan,               # L_std
                R_nan,                # R_dist
                np.nan,               # n_mom
                v_nan,                # v_mom
                np.nan,               # T_par_mom
                np.nan,               # T_perp_mom
                B_nan,                # B
                np.nan,               # qf
                success_nan,          # success
                counts_nan            # counts_save
            )
            results.append(dummy_res)

    if not results:
        print("⚠ No results to save.")
        return

    # Sort results based on timestamp
    results = sorted(results, key=lambda r: r[1].timestamp())

    # Unpack results into arrays
    (
        inds, t_vdfs, opt_params_all, param_names_all, in_conds_all, 
        constraint_flag_all, overlap_3d_all, ks_val_all, p_val_all, D_norm_all, L_mean_all, L_std_all, R_all, 
        n_all, v_bulk_bf_all, T_par_all, T_perp_all, B_all, qf_all, success_all, counts_all
    ) = zip(*results)

    # Convert to numpy arrays for saving
    times = np.array([t.timestamp() for t in t_vdfs], dtype=np.float64)[:, None]
    opt_params = np.stack(opt_params_all, axis=0)
    in_conds = np.stack(in_conds_all, axis=0)
    constraints = np.stack(constraint_flag_all, axis=0)
    overlap = np.array(overlap_3d_all)[:, None]
    ks_metric = np.stack([ks_val_all, p_val_all], axis=1)
    D_metric = np.stack([D_norm_all, L_mean_all, L_std_all], axis=1)
    R_dist = np.array(R_all)[:, None]
    n_mom = np.array(n_all)[:, None]
    v_mom = np.stack(v_bulk_bf_all, axis=0)
    T_mom = np.stack([T_par_all, T_perp_all], axis=1)
    B = np.stack(B_all, axis=0)
    qf = np.array(qf_all)[:, None]
    success = np.stack(success_all, axis=0)
    counts_save = np.stack(counts_all, axis=0)

    start_time = datetime.now()

    # Save all results at once
    save_all_results_h5(
        h5_path=h5_path,
        times=times,
        opt_params=opt_params,
        param_names=param_names_all[0],  # assume constant across runs
        in_conds=in_conds,
        constraints=constraints,
        overlap=overlap,
        ks_metric=ks_metric,
        D_metric=D_metric,
        R=R_dist,
        n_mom=n_mom,     
        v_mom=v_mom,         
        T_mom=T_mom,          
        B=B,
        qf=qf,
        success=success,
        counts_save=counts_save           
    )
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
           
def fit_data_h5(pick_model, t_vdf, vdf_in, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, vels, theta, R_dist, qf, method_in = 'powell', file_name = f'results_powell_cbc_par_drift_TEST', n_workers=32):
    """
    CHOOSE THE FITTING MODEL, FIT AND SAVE DATA!
    pick_model determines which fitting model to use.
    --------------------------------------------------------------------------------------------------------------------
    pick_model = 0 - Separate core and beam fit, only parallel drift, pick beam direction automatically!
    pick model = 1 - Core only Bi-Maxwellian fit!
     --------------------------------------------------------------------------------------------------------------------
    """

    N = len(vdf_in)
    theta_copies = [theta.copy() for _ in range(N)]

    if pick_model == 0:
        "*********************************************************************"
        "Separate core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_core_sep_par_drift_h5
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = file_name
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        core_only = False
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i], R_dist[i], qf[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 1:
        "*********************************************************************"
        "Core only!"
        "********************************************************************"
        fit_in = fit_one_core_only_h5
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = file_name
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        core_only = True
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i], R_dist[i], qf[i]) for i in range(N)]        
        "********************************************************************"
        
    run_serial_h5(tasks_in, fit_in, h5_path=save_path, core_only=core_only)

def load_results_h5(filename, start_time=None, end_time=None):
    """
    Load results from an HDF5 file, optionally filtering by time range.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    start_time : datetime, optional
        Start of time range filter (inclusive).
    end_time : datetime, optional
        End of time range filter (inclusive).

    Returns
    -------
    data : dict
        Dictionary with keys:
        - "time": np.ndarray of datetime objects
        - All other datasets in the file as np.ndarrays
    """
    with h5py.File(filename, "r") as f:
        # Convert timestamps to datetime
        times = np.array(list(map(datetime.fromtimestamp, f["time"][...].ravel())))        
        # Create mask if filtering
        if start_time or end_time:
            mask = np.ones(len(times), dtype=bool)
            if start_time:
                mask &= times >= start_time
            if end_time:
                mask &= times <= end_time
        else:
            mask = slice(None)  # No filtering

        # Load datasets
        data = {"time": times[mask]}
        for key in f.keys():
            if key == "time":
                continue
            data[key] = f[key][mask]

        if "fit_parameters" in f:
            param_names = f["fit_parameters"].attrs.get("param_names", None)
            if param_names is not None:
                param_names = [p.decode() if isinstance(p, bytes) else p for p in param_names]
                data["param_names"] = param_names

    return data

def format_data_h5(data):
    """
    Format loaded HDF5 data into structured arrays for analysis fore core + beam fits.
    """
    t_vdf = data['time']
    fitted_params = data['fit_parameters']
    data_transposed = np.array([col for col in data['fit_parameters'].T])
    nc_all, vxc, vyc, vzc, vth_par_c_all, vth_perp_c_all, nb_all, vxb, vyb, vzb, vth_par_b_all, vth_perp_b_all = data_transposed
    nc_all = nc_all * 1e-6 # in cm^-3
    nb_all = nb_all * 1e-6 # in cm^-3
    vc_all = np.stack((vxc, vyc, vzc), axis = 1) * 1e-3 # in km/s
    vb_all = np.stack((vxb, vyb, vzb), axis = 1) * 1e-3 # in km/s
    in_conds = data['initial_conditions']
    constraint_flag_all = data['constraints']
    overlap_all_3d = data['overlap'].T[0]
    ks_all = data['ks_metric']
    D_all = data['D_metric']
    R_all = data['R'].T[0]
    n_all = data['n_mom'].T[0] # in cm^-3
    v_all = data['v_mom'] * 1e-3 # in km/s
    T_par_all, T_perp_all = data['T_mom'][:, 0], data['T_mom'][:, 1]
    B_all = data['B'] # in nT
    qf_all = data['qf'].T[0]
    success_all = data['success']
    counts_all = data['counts_save']
    # fitted_params = data_transposed.T

    return nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, fitted_params, in_conds, constraint_flag_all, overlap_all_3d, ks_all, D_all, t_vdf, R_all, n_all, v_all, T_par_all, T_perp_all, B_all, qf_all, success_all, counts_all

def format_data_h5_core(data):
    """
    Format loaded HDF5 data into structured arrays for analysis for core only fits.
    """
    t_vdf = data['time']
    fitted_params = data['fit_parameters']
    data_transposed = np.array([col for col in data['fit_parameters'].T])
    nc_all, vxc, vyc, vzc, vth_par_c_all, vth_perp_c_all = data_transposed
    nc_all = nc_all * 1e-6 # in cm^-3
    vc_all = np.stack((vxc, vyc, vzc), axis = 1) * 1e-3 # in km/s
    in_conds = data['initial_conditions']
    constraint_flag_all = data['constraints']
    overlap_all_3d = data['overlap'].T[0]
    ks_all = data['ks_metric']
    D_all = data['D_metric']
    R_all = data['R'].T[0]
    n_all = data['n_mom'].T[0] # in cm^-3
    v_all = data['v_mom'] * 1e-3 # in km/s
    T_par_all, T_perp_all = data['T_mom'][:, 0], data['T_mom'][:, 1]
    B_all = data['B'] # in nT
    qf_all = data['qf'].T[0]
    success_all = data['success']
    counts_all = data['counts_save']
    # fitted_params = data_transposed.T

    return nc_all, vc_all, vth_par_c_all, vth_perp_c_all, fitted_params, in_conds, constraint_flag_all, overlap_all_3d, ks_all, D_all, t_vdf, R_all, n_all, v_all, T_par_all, T_perp_all, B_all, qf_all, success_all, counts_all
