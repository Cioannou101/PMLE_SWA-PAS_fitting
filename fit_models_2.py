
import os
import numpy as np
import lmfit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import scipy.constants as sc
from Poisson_fit_functions_2 import *
from gen_funcs_2 import *
from gof_funcs import *
from datetime import datetime
import multiprocessing as mp
import h5py

"Separate core and beam fit - parallelised version"
def fit_one_core_sep_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, vxb_init, theta = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    uxt, uyt, uzt = vx_bf, vy_bf, vz_bf
    umag = np.linalg.norm([uxt, uyt, uzt], axis = 0)
    # upar = uxt - v_bulk_bf[0] 
    # uperp = np.linalg.norm([uyt - v_bulk_bf[1], uzt - v_bulk_bf[2]])
    
    counts_core = np.copy(counts_in)
    # core_mask = upar < 0

    if vxb_init == -1.0:
        core_mask = umag > np.linalg.norm(v_bulk_bf)
    else:    
        core_mask = umag < np.linalg.norm(v_bulk_bf)
    counts_core[~core_mask] = np.nan
    
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
    
    constraints_max[0] = 1.5 * n_in
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
    
    n_beam = 0.18 * n_in
    vx_beam = vx_fit_c + (vxb_init * np.sign(vx_fit_c) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    # Beam Constraints
    constraints_min[6] = 0
    constraints_min[7] = vx_beam - 0.75*VA_in
    constraints_min[8] = vy_in - 50e3
    constraints_min[9] = vz_in - 50e3
    constraints_min[10] = 0.5 * vth_par_beam
    constraints_min[11] = 0.5 * vth_perp_beam
    
    constraints_max[6] = 0.5 * n_in
    constraints_max[7] = vx_beam + 0.75*VA_in
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
    fit_params.add('vy1', value=vy_fit_c, vary = False)
    fit_params.add('vz1', value=vz_fit_c,  vary = False)
    fit_params.add('v_th_par1', value=vth_par_fit_c, vary = False)
    fit_params.add('v_th_perp1', value=vth_perp_fit_c, vary = False)
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('vy2', value=vy_in, min=constraints_min[8], max=constraints_max[8])
    fit_params.add('vz2', value=vz_in, min=constraints_min[9], max=constraints_max[9])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])
    
    # Minimise
    minimiser_beam = lmfit.minimize(
        logp_double_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    
    # Extract optimised parameters
    opt_params_beam = np.array([minimiser_beam.params[name].value for name in minimiser_beam.var_names])
    
    (
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b
    ) = opt_params_beam
    
    "Now fit core again with set beam"
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_fit_c, min=constraints_min[0], max=constraints_max[0])
    fit_params.add('vx1', value=vx_fit_c, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy1', value=vy_fit_c, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz1', value=vz_fit_c, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par1', value=vth_par_fit_c, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp1', value=vth_perp_fit_c, min=constraints_min[5], max=constraints_max[5])
    
    fit_params.add('n_var2', value=n_fit_b, vary = False)
    fit_params.add('vx2', value=vx_fit_b, vary = False)
    fit_params.add('vy2', value=vy_fit_b, vary = False)
    fit_params.add('vz2', value=vz_fit_b, vary = False)
    fit_params.add('v_th_par2', value=vth_par_fit_b, vary = False)
    fit_params.add('v_th_perp2', value=vth_perp_fit_b, vary = False)
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_double_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    opt_params = np.zeros(len(minimiser_core.params))
    for i, j in enumerate(minimiser_core.params):
        opt_params[i] = minimiser_core.params[j].value
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params) | (constraints_min == opt_params)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_b, vz_fit_b]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_core, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

"core and beam together fit - parallelised version"
def fit_one_both_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels = tasks

    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel
    
    data_f = np.copy(vdf_in)
    data_integrated = integrate_vdf_over_angles(data_f, theta)
    ind_max = np.nanargmax(data_integrated)

    if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
        vxb_init = -1.0

    elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
        vxb_init = 1.0

    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_in, G_factors)
    
    # Velocities
    ux, uy, uz = vx_bf, vy_bf, vz_bf
    
    n_in = n * 1e6
    T_par_in, T_perp_in, T_in = T_par, T_perp, T
    # vth_par_in = np.sqrt(2 * T_par_in * sc.e / sc.m_p)
    vth_perp_in = np.sqrt(2 * T_perp_in * sc.e / sc.m_p)
    # vth_in = np.sqrt(2 * T_in * sc.e / sc.m_p)
    vx_in, vy_in, vz_in = v_bulk_bf
    B_in = B_all * 1e-9
    VA_in = np.linalg.norm(B_in) / np.sqrt(sc.mu_0 * sc.m_p * n_in)
    
    # Initial fit parameters
    n_core = 0.8 * n_in
    vx_core = vx_in * 0.95
    vth_par_core = vth_perp_in * 0.85
    vth_perp_core = vth_perp_in

    n_beam = 0.2 * n_in
    vx_beam = vx_in + (vxb_init * np.sign(vx_in) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    constraints_min = np.zeros(12)
    constraints_max = np.zeros(12)
    
    # Core Constraints
    constraints_min[0] = 0.5 * n_in
    constraints_min[1] = vx_core - VA_in
    constraints_min[2] = vy_in - 50e3
    constraints_min[3] = vz_in - 50e3
    constraints_min[4] = 0.5 * vth_par_core
    constraints_min[5] = 0.5 * vth_perp_core
    
    constraints_max[0] = nc_init * n_in
    constraints_max[1] = vx_core + VA_in
    constraints_max[2] = vy_in + 50e3
    constraints_max[3] = vz_in + 50e3
    constraints_max[4] = 2 * vth_par_core
    constraints_max[5] = 2 * vth_perp_core

    # Beam Constraints
    constraints_min[6] = 0
    constraints_min[7] = vx_beam - 1*VA_in
    constraints_min[8] = vy_in - 50e3
    constraints_min[9] = vz_in - 50e3
    constraints_min[10] = 0.5 * vth_par_beam
    constraints_min[11] = 0.5 * vth_perp_beam
    
    constraints_max[6] = 0.5 * n_in
    constraints_max[7] = vx_beam + 1*VA_in
    constraints_max[8] = vy_in + 50e3
    constraints_max[9] = vz_in + 50e3
    constraints_max[10] = 2 * vth_par_beam
    constraints_max[11] = 2 * vth_perp_beam
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_core, min=constraints_min[0], max=constraints_max[0])
    fit_params.add('vx1', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy1', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz1', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par1', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp1', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('vy2', value=vy_in, min=constraints_min[8], max=constraints_max[8])
    fit_params.add('vz2', value=vz_in, min=constraints_min[9], max=constraints_max[9])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])

    # method = 'powell'
    # method = 'differential_evolution'
    
    # Minimise
    minimiser = lmfit.minimize(
        logp_double_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
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

    opt_params = np.zeros(len(minimiser.params))
    for i, j in enumerate(minimiser.params):
        opt_params[i] = minimiser.params[j].value
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params) | (constraints_min == opt_params)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_b, vz_fit_b]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

"core and beam together fit, only paraller drift - parallelised version"
def fit_one_both_par_drift_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels = tasks

    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel
    
    data_f = np.copy(vdf_in)
    data_integrated = integrate_vdf_over_angles(data_f, theta)
    ind_max = np.nanargmax(data_integrated)

    if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
        vxb_init = -1.0

    elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
        vxb_init = 1.0

    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_in, G_factors)
    
    # Velocities
    ux, uy, uz = vx_bf, vy_bf, vz_bf
    
    n_in = n * 1e6
    T_par_in, T_perp_in, T_in = T_par, T_perp, T
    # vth_par_in = np.sqrt(2 * T_par_in * sc.e / sc.m_p)
    vth_perp_in = np.sqrt(2 * T_perp_in * sc.e / sc.m_p)
    # vth_in = np.sqrt(2 * T_in * sc.e / sc.m_p)
    vx_in, vy_in, vz_in = v_bulk_bf
    B_in = B_all * 1e-9
    VA_in = np.linalg.norm(B_in) / np.sqrt(sc.mu_0 * sc.m_p * n_in)
    
    # Initial fit parameters
    n_core = 0.8 * n_in
    vx_core = vx_in * 0.95
    vth_par_core = vth_perp_in * 0.85
    vth_perp_core = vth_perp_in

    n_beam = 0.2 * n_in
    vx_beam = vx_in + (vxb_init * np.sign(vx_in) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    constraints_min = np.zeros(12)
    constraints_max = np.zeros(12)
    
    # Core Constraints
    constraints_min[0] = 0.5 * n_in
    constraints_min[1] = vx_core - VA_in
    constraints_min[2] = vy_in - 50e3
    constraints_min[3] = vz_in - 50e3
    constraints_min[4] = 0.5 * vth_par_core
    constraints_min[5] = 0.5 * vth_perp_core
    
    constraints_max[0] = nc_init * n_in
    constraints_max[1] = vx_core + VA_in
    constraints_max[2] = vy_in + 50e3
    constraints_max[3] = vz_in + 50e3
    constraints_max[4] = 2 * vth_par_core
    constraints_max[5] = 2 * vth_perp_core

    # Beam Constraints
    constraints_min[6] = 0
    constraints_min[7] = vx_beam - 1*VA_in
    constraints_min[8] = vy_in - 50e3
    constraints_min[9] = vz_in - 50e3
    constraints_min[10] = 0.5 * vth_par_beam
    constraints_min[11] = 0.5 * vth_perp_beam
    
    constraints_max[6] = 0.5 * n_in
    constraints_max[7] = vx_beam + 1*VA_in
    constraints_max[8] = vy_in + 50e3
    constraints_max[9] = vz_in + 50e3
    constraints_max[10] = 2 * vth_par_beam
    constraints_max[11] = 2 * vth_perp_beam
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_core, min=constraints_min[0], max=constraints_max[0])
    fit_params.add('vx1', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par1', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp1', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])

    # method = 'powell'
    # method = 'differential_evolution'
    
    # Minimise
    minimiser = lmfit.minimize(
        logp_double_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
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

    opt_params = np.zeros(len(minimiser.params))
    for i, j in enumerate(minimiser.params):
        opt_params[i] = minimiser.params[j].value
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params
    opt_params_new = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

"Separate core, beam with repeats - parallelised version"
def fit_one_core_sep_parallel_repeats(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    ind, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, vxb_init, theta, n_repeats, tol_rep = tasks
    
    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    uxt, uyt, uzt = vx_bf, vy_bf, vz_bf
    umag = np.linalg.norm([uxt, uyt, uzt], axis = 0)
    
    counts_core = np.copy(counts_in)
    
    if vxb_init == -1.0:
        core_mask = umag > np.linalg.norm(v_bulk_bf)
    else:    
        core_mask = umag < np.linalg.norm(v_bulk_bf)
    
    counts_core[~core_mask] = np.nan
    
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_core, G_factors)
    
    # Velocities
    ux, uy, uz = vx_bf, vy_bf, vz_bf
    
    n_in = n * 1e6
    T_par_in, T_perp_in, T_in = T_par, T_perp, T
    # vth_par_in = np.sqrt(2 * T_par_in * sc.e / sc.m_p)
    vth_perp_in = np.sqrt(2 * T_perp_in * sc.e / sc.m_p)
    # vth_in = np.sqrt(2 * T_in * sc.e / sc.m_p)
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
    
    constraints_max[0] = 1.5 * n_in
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
    
    n_beam = 0.18 * n_in
    vx_beam = vx_fit_c + (vxb_init * np.sign(vx_fit_c) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    # Beam Constraints
    constraints_min[6] = 0
    constraints_min[7] = vx_beam - 1*VA_in
    constraints_min[8] = vy_in - 50e3
    constraints_min[9] = vz_in - 50e3
    constraints_min[10] = 0.5 * vth_par_beam
    constraints_min[11] = 0.5 * vth_perp_beam
    
    constraints_max[6] = 0.5 * n_in
    constraints_max[7] = vx_beam + 1*VA_in
    constraints_max[8] = vy_in + 50e3
    constraints_max[9] = vz_in + 50e3
    constraints_max[10] = 2 * vth_par_beam
    constraints_max[11] = 2 * vth_perp_beam
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_fit_c, vary = False)
    fit_params.add('vx1', value=vx_fit_c, vary = False)
    fit_params.add('vy1', value=vy_fit_c, vary = False)
    fit_params.add('vz1', value=vz_fit_c,  vary = False)
    fit_params.add('v_th_par1', value=vth_par_fit_c, vary = False)
    fit_params.add('v_th_perp1', value=vth_perp_fit_c, vary = False)
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('vy2', value=vy_in, min=constraints_min[8], max=constraints_max[8])
    fit_params.add('vz2', value=vz_in, min=constraints_min[9], max=constraints_max[9])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])
    
    # Minimise
    minimiser_beam = lmfit.minimize(
        logp_double_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    # Extract optimised parameters
    opt_params_beam = np.array([minimiser_beam.params[name].value for name in minimiser_beam.var_names])
    
    (
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b
    ) = opt_params_beam
    
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

    "Now fit core again with set beam"
    def repeated_core_beam_fit(
    ux, uy, uz, dt, G_in, counts_data,
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b,
    constraints_min, constraints_max,
    method_func='powell', n_repeats=1, tol=0.01
):
        """
        Repeatedly fit core and beam components until convergence or max iterations.
        Terminates early if all parameter changes are <1% from previous iteration.
        """
        # Initial parameters
        params_old = np.array([
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b
        ])

        for _ in range(n_repeats):
            # --- Fit Core with Beam Fixed ---
            fit_params = lmfit.Parameters()
            # Core parameters (varying)
            fit_params.add('n_var1', value=params_old[0], min=constraints_min[0], max=constraints_max[0])
            fit_params.add('vx1', value=params_old[1], min=constraints_min[1], max=constraints_max[1])
            fit_params.add('vy1', value=params_old[2], min=constraints_min[2], max=constraints_max[2])
            fit_params.add('vz1', value=params_old[3], min=constraints_min[3], max=constraints_max[3])
            fit_params.add('v_th_par1', value=params_old[4], min=constraints_min[4], max=constraints_max[4])
            fit_params.add('v_th_perp1', value=params_old[5], min=constraints_min[5], max=constraints_max[5])
            # Beam parameters (fixed)
            fit_params.add('n_var2', value=params_old[6], vary=False)
            fit_params.add('vx2', value=params_old[7], vary=False)
            fit_params.add('vy2', value=params_old[8], vary=False)
            fit_params.add('vz2', value=params_old[9], vary=False)
            fit_params.add('v_th_par2', value=params_old[10], vary=False)
            fit_params.add('v_th_perp2', value=params_old[11], vary=False)

            minimiser_core = lmfit.minimize(
                logp_double_minimisation,
                fit_params,
                method=method_func,
                args=(ux, uy, uz, G_in, dt, counts_data)
            )

            params_mid = np.array([minimiser_core.params[key].value for key in minimiser_core.params])

            # --- Fit Beam with Core Fixed ---
            fit_params = lmfit.Parameters()
            # Core parameters (fixed)
            fit_params.add('n_var1', value=params_mid[0], vary=False)
            fit_params.add('vx1', value=params_mid[1], vary=False)
            fit_params.add('vy1', value=params_mid[2], vary=False)
            fit_params.add('vz1', value=params_mid[3], vary=False)
            fit_params.add('v_th_par1', value=params_mid[4], vary=False)
            fit_params.add('v_th_perp1', value=params_mid[5], vary=False)
            # Beam parameters (varying)
            fit_params.add('n_var2', value=params_mid[6], min=constraints_min[6], max=constraints_max[6])
            fit_params.add('vx2', value=params_mid[7], min=constraints_min[7], max=constraints_max[7])
            fit_params.add('vy2', value=params_mid[8], min=constraints_min[8], max=constraints_max[8])
            fit_params.add('vz2', value=params_mid[9], min=constraints_min[9], max=constraints_max[9])
            fit_params.add('v_th_par2', value=params_mid[10], min=constraints_min[10], max=constraints_max[10])
            fit_params.add('v_th_perp2', value=params_mid[11], min=constraints_min[11], max=constraints_max[11])

            minimiser_beam = lmfit.minimize(
                logp_double_minimisation,
                fit_params,
                method=method_func,
                args=(ux, uy, uz, G_in, dt, counts_data)
            )

            params_new = np.array([minimiser_beam.params[key].value for key in minimiser_beam.params])

            # --- Termination condition: all params change < 1% ---
            relative_changes = np.abs((params_new - params_old) / np.where(params_old != 0, params_old, 1))
            if np.all(relative_changes < tol):
                break

            params_old = params_new.copy()

        # Unpack results
        return minimiser_beam, params_new


    minimiser_final, opt_params = repeated_core_beam_fit(
    ux, uy, uz, dt, G_in, counts_data,
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b,
    constraints_min, constraints_max,
    method_func=method,
    n_repeats = n_repeats,
    tol = tol_rep
)

    (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b) = opt_params

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = np.isclose(opt_params, constraints_min, rtol=1e-05, atol=1e-08) | np.isclose(opt_params, constraints_max, rtol=1e-05, atol=1e-08)
    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_b, vz_fit_b]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, vth_par_fit_c, vth_perp_fit_c, n_fit_b, vb, vth_par_fit_b, vth_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_final, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

"Separate core, beam, core, beam, core fits - parallelised version"
def fit_one_core_sep_parallel_repeats_segregated(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    ind, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, vxb_init, theta, n_repeats, tol_rep = tasks
    
    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    umag = np.linalg.norm([vx_bf, vy_bf, vz_bf], axis = 0)
    counts_core = np.copy(counts_in)
    core_mask = umag < np.linalg.norm(v_bulk_bf)
    counts_core[~core_mask] = np.nan
    
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_core, G_factors)
    
    # Velocities
    ux, uy, uz = vx_bf, vy_bf, vz_bf
    
    n_in = n * 1e6
    T_par_in, T_perp_in, T_in = T_par, T_perp, T
    # vth_par_in = np.sqrt(2 * T_par_in * sc.e / sc.m_p)
    vth_perp_in = np.sqrt(2 * T_perp_in * sc.e / sc.m_p)
    # vth_in = np.sqrt(2 * T_in * sc.e / sc.m_p)
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
    
    constraints_max[0] = 1.5 * n_in
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
    
    # results_core.append(minimiser_core)
    
    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])
    
    (
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    ) = opt_params_core
    
    
    "Now fit beam with set core"
    umag = np.linalg.norm([vx_bf, vy_bf, vz_bf], axis = 0)
    counts_beam = np.copy(counts_in)
    beam_mask = umag > np.linalg.norm(v_bulk_bf)
    counts_beam[~beam_mask] = np.nan
    
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_beam, G_factors)
    
    # Initial fit parameters
    
    n_beam = 0.18 * n_in
    vx_beam = vx_fit_c + (vxb_init * np.sign(vx_fit_c) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    # Beam Constraints
    constraints_min[6] = 0
    constraints_min[7] = vx_beam - 1*VA_in
    constraints_min[8] = vy_in - 50e3
    constraints_min[9] = vz_in - 50e3
    constraints_min[10] = 0.5 * vth_par_beam
    constraints_min[11] = 0.5 * vth_perp_beam
    
    constraints_max[6] = 0.5 * n_in
    constraints_max[7] = vx_beam + 1*VA_in
    constraints_max[8] = vy_in + 50e3
    constraints_max[9] = vz_in + 50e3
    constraints_max[10] = 2 * vth_par_beam
    constraints_max[11] = 2 * vth_perp_beam
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_fit_c, vary = False)
    fit_params.add('vx1', value=vx_fit_c, vary = False)
    fit_params.add('vy1', value=vy_fit_c, vary = False)
    fit_params.add('vz1', value=vz_fit_c,  vary = False)
    fit_params.add('v_th_par1', value=vth_par_fit_c, vary = False)
    fit_params.add('v_th_perp1', value=vth_perp_fit_c, vary = False)
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('vy2', value=vy_in, min=constraints_min[8], max=constraints_max[8])
    fit_params.add('vz2', value=vz_in, min=constraints_min[9], max=constraints_max[9])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])
    
    # Minimise
    minimiser_beam = lmfit.minimize(
        logp_double_minimisation,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    # Extract optimised parameters
    opt_params_beam = np.array([minimiser_beam.params[name].value for name in minimiser_beam.var_names])
    
    (
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b
    ) = opt_params_beam
    
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

    "Now fit core again with set beam"
    def repeated_core_beam_fit(
    ux, uy, uz, dt, G_in, counts_in,
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b,
    constraints_min, constraints_max,
    method_func='powell', n_repeats=1, tol=0.01
):
        """
        Repeatedly fit core and beam components until convergence or max iterations.
        Terminates early if all parameter changes are <1% from previous iteration.
        """
        # Initial parameters
        params_old = np.array([
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b
        ])

        for _ in range(n_repeats):
            
            # --- Fit Core with Beam Fixed ---
            umag = np.linalg.norm([vx_bf, vy_bf, vz_bf], axis = 0)
            counts_core = np.copy(counts_in)
            core_mask = umag < (np.linalg.norm(v_bulk_bf) + 40e3)
            counts_core[~core_mask] = np.nan
            
            # Handle invalid G values: set NaNs where G is invalid
            counts_data, G_in = clean_inputs(counts_core, G_factors)

            fit_params = lmfit.Parameters()
            # Core parameters (varying)
            fit_params.add('n_var1', value=params_old[0], min=constraints_min[0], max=constraints_max[0])
            fit_params.add('vx1', value=params_old[1], min=constraints_min[1], max=constraints_max[1])
            fit_params.add('vy1', value=params_old[2], min=constraints_min[2], max=constraints_max[2])
            fit_params.add('vz1', value=params_old[3], min=constraints_min[3], max=constraints_max[3])
            fit_params.add('v_th_par1', value=params_old[4], min=constraints_min[4], max=constraints_max[4])
            fit_params.add('v_th_perp1', value=params_old[5], min=constraints_min[5], max=constraints_max[5])
            # Beam parameters (fixed)
            fit_params.add('n_var2', value=params_old[6], vary=False)
            fit_params.add('vx2', value=params_old[7], vary=False)
            fit_params.add('vy2', value=params_old[8], vary=False)
            fit_params.add('vz2', value=params_old[9], vary=False)
            fit_params.add('v_th_par2', value=params_old[10], vary=False)
            fit_params.add('v_th_perp2', value=params_old[11], vary=False)

            minimiser_core = lmfit.minimize(
                logp_double_minimisation,
                fit_params,
                method=method_func,
                args=(ux, uy, uz, G_in, dt, counts_data)
            )

            params_mid = np.array([minimiser_core.params[key].value for key in minimiser_core.params])

            # --- Fit Beam with Core Fixed ---
            umag = np.linalg.norm([vx_bf, vy_bf, vz_bf], axis = 0)
            counts_beam = np.copy(counts_in)
            beam_mask = umag > (np.linalg.norm(v_bulk_bf) -40e3)
            counts_beam[~beam_mask] = np.nan
            
            # Handle invalid G values: set NaNs where G is invalid
            counts_data, G_in = clean_inputs(counts_beam, G_factors)

            fit_params = lmfit.Parameters()
            # Core parameters (fixed)
            fit_params.add('n_var1', value=params_mid[0], vary=False)
            fit_params.add('vx1', value=params_mid[1], vary=False)
            fit_params.add('vy1', value=params_mid[2], vary=False)
            fit_params.add('vz1', value=params_mid[3], vary=False)
            fit_params.add('v_th_par1', value=params_mid[4], vary=False)
            fit_params.add('v_th_perp1', value=params_mid[5], vary=False)
            # Beam parameters (varying)
            fit_params.add('n_var2', value=params_mid[6], min=constraints_min[6], max=constraints_max[6])
            fit_params.add('vx2', value=params_mid[7], min=constraints_min[7], max=constraints_max[7])
            fit_params.add('vy2', value=params_mid[8], min=constraints_min[8], max=constraints_max[8])
            fit_params.add('vz2', value=params_mid[9], min=constraints_min[9], max=constraints_max[9])
            fit_params.add('v_th_par2', value=params_mid[10], min=constraints_min[10], max=constraints_max[10])
            fit_params.add('v_th_perp2', value=params_mid[11], min=constraints_min[11], max=constraints_max[11])

            minimiser_beam = lmfit.minimize(
                logp_double_minimisation,
                fit_params,
                method=method_func,
                args=(ux, uy, uz, G_in, dt, counts_data)
            )

            params_new = np.array([minimiser_beam.params[key].value for key in minimiser_beam.params])

            # --- Termination condition: all params change < 1% ---
            relative_changes = np.abs((params_new - params_old) / np.where(params_old != 0, params_old, 1))
            if np.all(relative_changes < tol):
                break

            params_old = params_new.copy()

        # Unpack results
        return minimiser_beam, params_new


    minimiser_final, opt_params = repeated_core_beam_fit(
    ux, uy, uz, dt, G_in, counts_in,
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b,
    constraints_min, constraints_max,
    method_func=method,
    n_repeats = n_repeats,
    tol = tol_rep
)

    (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b) = opt_params

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = np.isclose(opt_params, constraints_min, rtol=1e-05, atol=1e-08) | np.isclose(opt_params, constraints_max, rtol=1e-05, atol=1e-08)
    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, vth_par_fit_b, vth_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_b, vz_fit_b]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, vth_par_fit_c, vth_perp_fit_c, n_fit_b, vb, vth_par_fit_b, vth_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_final, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_one_core_sep_par_drift_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    if np.all((counts_in == 0) | np.isnan(counts_in)):

        return ind, np.nan, np.nan, np.full(12, np.nan), np.full(12, np.nan), np.full(12, np.nan), np.nan, np.nan, np.nan

    try:
        uxt, uyt, uzt = vx_bf, vy_bf, vz_bf
        umag = np.linalg.norm([uxt, uyt, uzt], axis = 0)
        # upar = uxt - v_bulk_bf[0] 
        # uperp = np.linalg.norm([uyt - v_bulk_bf[1], uzt - v_bulk_bf[2]])
        
        counts_core = np.copy(counts_in)
        # core_mask = upar < 0
        # print(counts_core)
        data_f = np.copy(vdf_in)
        data_integrated = integrate_vdf_over_angles(data_f, theta)
        ind_max = np.nanargmax(data_integrated)

        if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
            core_mask = umag > np.linalg.norm(v_bulk_bf)
            vxb_init = -1.0

        elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
            core_mask = umag < np.linalg.norm(v_bulk_bf)
            vxb_init = 1.0

        counts_core[~core_mask] = np.nan
    
    # if core data proccessing fails, return NaNs
    except Exception as e:
        print(f"Error processing counts_core at index {ind}: {e}")
        return ind, np.nan, np.nan, np.full(12, np.nan), np.full(12, np.nan), np.full(12, np.nan), np.nan, np.nan, np.nan

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
    
    constraints_max[0] = 1.5 * n_in
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
    
    n_beam = 0.18 * n_in
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
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    opt_params_new = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_core, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_one_sep_kappa_par_drift_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    uxt, uyt, uzt = vx_bf, vy_bf, vz_bf
    umag = np.linalg.norm([uxt, uyt, uzt], axis = 0)
    # upar = uxt - v_bulk_bf[0] 
    # uperp = np.linalg.norm([uyt - v_bulk_bf[1], uzt - v_bulk_bf[2]])
    
    counts_core = np.copy(counts_in)
    # core_mask = upar < 0

    data_f = np.copy(vdf_in)
    data_integrated = integrate_vdf_over_angles(data_f, theta)
    ind_max = np.nanargmax(data_integrated)

    if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
        core_mask = umag > np.linalg.norm(v_bulk_bf)
        vxb_init = -1.0

    elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
        core_mask = umag < np.linalg.norm(v_bulk_bf)
        vxb_init = 1.0

    counts_core[~core_mask] = np.nan
    
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
    kappa_core = 10.0
    constraints_kappa = [1.501, 150.0]
    
    constraints_min = np.zeros(12)
    constraints_max = np.zeros(12)
    
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
    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_core, min=constraints_min[0], max=nc_init * n_in)
    fit_params.add('vx', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    fit_params.add('kappa1', value=kappa_core, min=constraints_kappa[0], max = constraints_kappa[1])
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_minimisation_kappa,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    # results_core.append(minimiser_core)
    
    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])
    
    (
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c, kappa_c
    ) = opt_params_core
    
    "Now fit beam with set core"
    
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_in, G_factors)
    
    # Initial fit parameters
    
    n_beam = 0.18 * n_in
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
    fit_params.add('kappa1', value=kappa_c, vary = False)
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])
    
    # Minimise
    minimiser_beam = lmfit.minimize(
        logp_kappa_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
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
    fit_params.add('kappa1', value=kappa_c, min=constraints_kappa[0], max = constraints_kappa[1])

    fit_params.add('n_var2', value=n_fit_b, vary = False)
    fit_params.add('vx2', value=vx_fit_b, vary = False)
    fit_params.add('v_th_par2', value=vth_par_fit_b, vary = False)
    fit_params.add('v_th_perp2', value=vth_perp_fit_b, vary = False)
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_kappa_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    opt_params = np.zeros(len(minimiser_core.params))
    for i, j in enumerate(minimiser_core.params):
        opt_params[i] = minimiser_core.params[j].value
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    opt_params_new = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_kappa(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one_kappa(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_core, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_one_both_kappa_par_drift_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    data_f = np.copy(vdf_in)
    data_integrated = integrate_vdf_over_angles(data_f, theta)
    ind_max = np.nanargmax(data_integrated)

    if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
        vxb_init = -1.0

    elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
        vxb_init = 1.0
    
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
    n_core = 0.8 * n_in
    vx_core = vx_in * 0.95
    vth_par_core = vth_perp_in * 0.85
    vth_perp_core = vth_perp_in
    kappa_core = 10.0
    constraints_kappa = [1.501, 150.0]

    n_beam = 0.18 * n_in
    vx_beam = vx_in + (vxb_init * np.sign(vx_in) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in
    
    constraints_min = np.zeros(12)
    constraints_max = np.zeros(12)
    
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
    fit_params.add('n_var1', value=n_core, min=constraints_min[0], max=nc_init * n_in)
    fit_params.add('vx1', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par1', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp1', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    fit_params.add('kappa1', value=kappa_core, min=constraints_kappa[0], max = constraints_kappa[1])
    
    fit_params.add('n_var2', value=n_beam, min=constraints_min[6], max=constraints_max[6])
    fit_params.add('vx2', value=vx_beam, min=constraints_min[7], max=constraints_max[7])
    fit_params.add('v_th_par2', value=vth_par_beam, min=constraints_min[10], max=constraints_max[10])
    fit_params.add('v_th_perp2', value=vth_perp_beam, min=constraints_min[11], max=constraints_max[11])
    
    # Minimise
    minimiser = lmfit.minimize(
        logp_kappa_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    opt_params = np.zeros(len(minimiser.params))
    for i, j in enumerate(minimiser.params):
        opt_params[i] = minimiser.params[j].value
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    opt_params_new = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_kappa(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one_kappa(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_one_sep_kappa_beam_par_drift_parallel(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    uxt, uyt, uzt = vx_bf, vy_bf, vz_bf
    umag = np.linalg.norm([uxt, uyt, uzt], axis = 0)
    # upar = uxt - v_bulk_bf[0] 
    # uperp = np.linalg.norm([uyt - v_bulk_bf[1], uzt - v_bulk_bf[2]])
    
    counts_core = np.copy(counts_in)
    # core_mask = upar < 0

    data_f = np.copy(vdf_in)
    data_integrated = integrate_vdf_over_angles(data_f, theta)
    ind_max = np.nanargmax(data_integrated)

    if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
        core_mask = umag > np.linalg.norm(v_bulk_bf)
        vxb_init = -1.0

    elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
        core_mask = umag < np.linalg.norm(v_bulk_bf)
        vxb_init = 1.0

    counts_core[~core_mask] = np.nan
    
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
    
    constraints_max[0] = 1.5 * n_in
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
    
    # results_core.append(minimiser_core)
    
    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])
    
    (
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c
    ) = opt_params_core
    
    "Now fit beam with set core"
    
    # Handle invalid G values: set NaNs where G is invalid
    counts_data, G_in = clean_inputs(counts_in, G_factors)
    
    # Initial fit parameters
    
    n_beam = 0.18 * n_in
    vx_beam = vx_fit_c + (vxb_init * np.sign(vx_fit_c) * VA_in)
    vth_par_beam = vth_perp_in * 0.85
    vth_perp_beam = vth_perp_in

    kappa_beam = 10.0
    constraints_kappa = [1.501, 150.0]
    
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
    fit_params.add('kappa2', value=kappa_beam, min = constraints_kappa[0], max = constraints_kappa[1])
    
    # Minimise
    minimiser_beam = lmfit.minimize(
        logp_kappa_beam_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    # Extract optimised parameters
    opt_params_beam = np.array([minimiser_beam.params[name].value for name in minimiser_beam.var_names])
    
    (
    n_fit_b, vx_fit_b, vth_par_fit_b, vth_perp_fit_b, kappa_b
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
    fit_params.add('kappa2', value=kappa_b, vary = False)
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_kappa_beam_minimisation_par_drift,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    opt_params = np.zeros(len(minimiser_core.params))
    for i, j in enumerate(minimiser_core.params):
        opt_params[i] = minimiser_core.params[j].value
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b, kappa_b = opt_params

    opt_params_new = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_kappa(vx_bf, vy_bf, vz_bf,
                      n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b, kappa_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one_kappa_beam(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, kappa_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_core, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_one_core_only(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel
    
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
    n_core = 1 * n_in
    vx_core = vx_in
    vth_par_core = vth_perp_in * 0.95
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
    
    constraints_max[0] = 1.5 * n_in
    constraints_max[1] = vx_core + VA_in
    constraints_max[2] = vy_in + 50e3
    constraints_max[3] = vz_in + 50e3
    constraints_max[4] = 2 * vth_par_core
    constraints_max[5] = 2 * vth_perp_core

    # Initial Conditions
    in_conds = np.zeros(12)
    in_conds[0] = n_core
    in_conds[1] = vx_core
    in_conds[2] = vy_in
    in_conds[3] = vz_in
    in_conds[4] = vth_par_core
    in_conds[5] = vth_perp_core

    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_core, min=constraints_min[0], max=constraints_max[0])
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
    
    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])
    
    (
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c,
    ) = opt_params_core

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = np.zeros(12, dtype=bool)
    constraint_flag_all[:6] = (constraints_max[:6] == opt_params_core) | (constraints_min[:6] == opt_params_core)

    "Cor only. No overlap metric"
    overlap_3d = np.nan
    overlap_1d= np.nan

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one_core_only(n_fit_c, vc, vth_par_fit_c, vth_perp_fit_c, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_core, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_one_core_only_kappa(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel
    
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
    n_core = 1 * n_in
    vx_core = vx_in
    vth_par_core = vth_perp_in * 0.95
    vth_perp_core = vth_perp_in
    kappa_core = 10.0
    
    constraints_min = np.zeros(12)
    constraints_max = np.zeros(12)
    constraints_kappa = [1.501, 150.0]
    
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
    in_conds = np.zeros(12)
    in_conds[0] = n_core
    in_conds[1] = vx_core
    in_conds[2] = vy_in
    in_conds[3] = vz_in
    in_conds[4] = vth_par_core
    in_conds[5] = vth_perp_core

    
    # Define fit parameters
    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_core, min=constraints_min[0], max=constraints_max[0])
    fit_params.add('vx', value=vx_core, min=constraints_min[1], max=constraints_max[1])
    fit_params.add('vy', value=vy_in, min=constraints_min[2], max=constraints_max[2])
    fit_params.add('vz', value=vz_in, min=constraints_min[3], max=constraints_max[3])
    fit_params.add('v_th_par', value=vth_par_core, min=constraints_min[4], max=constraints_max[4])
    fit_params.add('v_th_perp', value=vth_perp_core, min=constraints_min[5], max=constraints_max[5])
    fit_params.add('kappa1', value=kappa_core, min=constraints_kappa[0], max = constraints_kappa[1])
    
    # Minimise
    minimiser_core = lmfit.minimize(
        logp_minimisation_kappa,
        fit_params,
        method=method,
        args=(ux, uy, uz, G_in, dt, counts_data),
        # tol=1e-3
    )
    
    # results_core.append(minimiser_core)
    
    # Extract optimised parameters
    opt_params_core = np.array([minimiser_core.params[name].value for name in minimiser_core.var_names])
    
    (
        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, vth_par_fit_c, vth_perp_fit_c, kappa_c,
    ) = opt_params_core

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = np.zeros(12, dtype=bool)
    constraint_flag_all[:6] = (constraints_max[:6] == opt_params_core[:6]) | (constraints_min[:6] == opt_params_core[:6])

    "Cor only. No overlap metric"
    overlap_3d = np.nan
    overlap_1d = np.nan

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one_core_only_kappa(n_fit_c, vc, vth_par_fit_c, vth_perp_fit_c, kappa_c, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    return (ind, minimiser_core, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_1d, overlap_3d, goodness_metric)

def fit_data(pick_model, t_vdf, vdf_in, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, vels, theta, date1_str, method_in = 'powell', long = False):
    """
    CHOOSE THE FITTING MODEL, FIT AND SAVE DATA!
    pick_model determines which fitting model to use.
    --------------------------------------------------------------------------------------------------------------------
    pick_model = 0 - Separate core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 1 - Core and beam fit together, only parallel drift, pick beam direction automatically!
    pick_model = 2 - Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 3 - Kappa core and beam fit together, only parallel drift, pick beam direction automatically!
    pick model = 4 - Core only Bi-Maxwellian fit!
    pick model = 5 - Core only Bi-kappa fit!
    pick_model = 6 - Separate bi_Max core and Kappa beam fit, only parallel drift, pick beam direction automatically!
     --------------------------------------------------------------------------------------------------------------------
    """

    N = len(vdf_in)
    theta_copies = [theta.copy() for _ in range(N)]

    if pick_model == 0:
        "*********************************************************************"
        "Separate core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_core_sep_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_cbc_par_drift'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 1:
        "*********************************************************************"
        "Core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_both_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_par_drift'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"
        
    if pick_model == 2:
        "*********************************************************************"
        "Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_sep_kappa_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_cbc_kappa'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 3:
        "*********************************************************************"
        "Kappa core and beam fit together, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_both_kappa_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_kappa'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"
        
    if pick_model == 4:
        "*********************************************************************"
        "Core only Bi-Maxwellian fit!"
        "********************************************************************"
        fit_in = fit_one_core_only
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        save_path = f'results_{method[0]}_core_only'
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 5:
        "*********************************************************************"
        "Core only Kappa fit!"
        "********************************************************************"
        fit_in = fit_one_core_only_kappa
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        save_path = f'results_{method[0]}_core_only_kappa'
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i]) for i in range(N)]
        "********************************************************************"
    
    if pick_model == 6:
        "*********************************************************************"
        "Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_sep_kappa_beam_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_cbc_kappa_beam'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"

    with ProcessPoolExecutor(max_workers=32) as executor:
            results = list(tqdm(executor.map(fit_in, tasks_in), total=N))

    results_fit = []
    in_conds = np.zeros([N, 12])
    constraints_min = np.zeros([N, 12])
    constraints_max = np.zeros([N, 12])
    constraint_flag_all = np.zeros([N, 12])
    overlap_all_1d = np.zeros(N)
    overlap_all_3d = np.zeros(N)
    goodness_metric = np.zeros(N)

    for i, (idx, minimiser, initial_conditions, min_con, max_con, constraint_flags_out, ov_1d, ov_3d, g_metric) in enumerate(results):
        results_fit.append(minimiser)
        in_conds[i] = initial_conditions
        constraints_min[i] = min_con 
        constraints_max[i] = max_con
        constraint_flag_all[i] = constraint_flags_out
        overlap_all_1d[i] = ov_1d
        overlap_all_3d[i] = ov_3d
        goodness_metric[i] = g_metric
        
    # Create the folder path using os.path.join for safety
    if long == True:
        folder_path = os.path.join('long_fit_results', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', save_path)
    else:
        folder_path = os.path.join('fit_results', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', save_path)

    os.makedirs(folder_path, exist_ok=True)

    # Save each file with full path
    file1 = os.path.join(folder_path, 'fit_results.npy')
    np.save(file1, results_fit)

    file2 = os.path.join(folder_path, 'initial_conditions.npy')
    np.save(file2, in_conds)

    constraints_all = np.array([constraints_min, constraints_max, constraint_flag_all])
    file3 = os.path.join(folder_path, 'constraints.npy')
    np.save(file3, constraints_all)

    overlap_save = np.array([overlap_all_1d, overlap_all_3d])
    file4 = os.path.join(folder_path, 'overlap.npy')
    np.save(file4, overlap_save)

    file5 = os.path.join(folder_path, 'goodness_metric.npy')
    np.save(file5, goodness_metric)

    file6 = os.path.join(folder_path, 'time.npy')
    np.save(file6, t_vdf)

def load_fitted_data(pick_model, date1_str, method_in = 'powell', long = False):
    """
    Load the fitted data!
    pick_model determines which fitting model to load.
    --------------------------------------------------------------------------------------------------------------------
    pick_model = 0 - Separate core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 1 - Core and beam fit together, only parallel drift, pick beam direction automatically!
    pick_model = 2 - Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 3 - Kappa core and beam fit together, only parallel drift, pick beam direction automatically!
    pick model = 4 - Core only Bi-Maxwellian fit!
    pick model = 5 - Core only Bi-kappa fit!
    pick_model = 6 - Separate bi_Max core and Kappa beam fit, only parallel drift, pick beam direction automatically!
     --------------------------------------------------------------------------------------------------------------------
    """

    if pick_model == 0:
        load_path = f'results_{method_in}_cbc_par_drift'
    if pick_model == 1:
        load_path = f'results_{method_in}_par_drift'
    if pick_model == 2:
        load_path = f'results_{method_in}_cbc_kappa'
    if pick_model == 3:
        load_path = f'results_{method_in}_kappa'
    if pick_model == 4:
        load_path = f'results_{method_in}_core_only'
    if pick_model == 5:
        load_path = f'results_{method_in}_core_only_kappa'
    if pick_model == 6:
        load_path = f'results_{method_in}_cbc_kappa_beam'
    
    if long == True:
        folder_path = os.path.join('long_fit_results', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', load_path)
    else:
        folder_path = os.path.join('fit_results', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', load_path)

    path2 = os.path.join(folder_path, 'initial_conditions.npy')
    in_conds = np.load(path2, allow_pickle=True)

    path3 = os.path.join(folder_path, 'constraints.npy')
    constraints_min, constraints_max, constraint_flag_all = np.load(path3, allow_pickle=True)

    path4 = os.path.join(folder_path, 'overlap.npy')
    overlap_all_1d, overlap_all_3d = np.load(path4, allow_pickle=True)

    path5 = os.path.join(folder_path, 'goodness_metric.npy')
    goodness_all = np.load(path5, allow_pickle=True)

    path = os.path.join(folder_path, 'fit_results.npy')
    results = np.load(path, allow_pickle=True)

    N = len(results)

    if pick_model == 0 or pick_model == 1:

        nc_all = np.zeros(N)
        vc_all = np.zeros([N, 3])
        vth_par_c_all = np.zeros(N)
        vth_perp_c_all = np.zeros(N)
        nb_all = np.zeros(N)
        vb_all = np.zeros([N, 3])
        vth_par_b_all = np.zeros(N)
        vth_perp_b_all = np.zeros(N)
        fitted_params = np.zeros((N, 12))

        for k in range(len(results)):
            try:
                minimiser = results[k]
                
                opt_params = np.zeros(len(minimiser.params))
                for i, j in enumerate(minimiser.params):
                    opt_params[i] = minimiser.params[j].value

                n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params
                vy_fit_b, vz_fit_b = vy_fit_c, vz_fit_c
                opt_params = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b

                nc_all[k] = n_fit_c
                vc_all[k] = np.array([vx_fit_c, vy_fit_c, vz_fit_c]) 
                vth_par_c_all[k] = v_th_par_fit_c
                vth_perp_c_all[k] = v_th_perp_fit_c
                nb_all[k] = n_fit_b
                vb_all[k] = np.array([vx_fit_b, vy_fit_b, vz_fit_b]) 
                vth_par_b_all[k] = v_th_par_fit_b
                vth_perp_b_all[k] = v_th_perp_fit_b
                fitted_params[k] = opt_params


            except Exception as e:
                print(f"Error processing result {k}: {e}")
                nc_all[k] = np.nan
                vc_all[k] = np.array([np.nan, np.nan, np.nan]) 
                vth_par_c_all[k] = np.nan
                vth_perp_c_all[k] = np.nan
                nb_all[k] = np.nan
                vb_all[k] = np.array([np.nan, np.nan, np.nan]) 
                vth_par_b_all[k] = np.nan
                vth_perp_b_all[k] = np.nan
                fitted_params[k] = np.full(12, np.nan)                
        
        return (nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, fitted_params, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_all_1d, overlap_all_3d, goodness_all)

    if pick_model == 2 or pick_model == 3:
        nc_all = np.zeros(N)
        vc_all = np.zeros([N, 3])
        vth_par_c_all = np.zeros(N)
        vth_perp_c_all = np.zeros(N)
        nb_all = np.zeros(N)
        vb_all = np.zeros([N, 3])
        vth_par_b_all = np.zeros(N)
        vth_perp_b_all = np.zeros(N)
        kappa_all = np.zeros(N)
        fitted_params = np.zeros((N, 12))

        for k in range(len(results)):
            
            minimiser = results[k]
            
            opt_params = np.zeros(len(minimiser.params))
            for i, j in enumerate(minimiser.params):
                opt_params[i] = minimiser.params[j].value

            n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params
            vy_fit_b, vz_fit_b = vy_fit_c, vz_fit_c
            opt_params = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b 

            nc_all[k] = n_fit_c
            vc_all[k] = np.array([vx_fit_c, vy_fit_c, vz_fit_c]) 
            vth_par_c_all[k] = v_th_par_fit_c
            vth_perp_c_all[k] = v_th_perp_fit_c
            nb_all[k] = n_fit_b
            vb_all[k] = np.array([vx_fit_b, vy_fit_b, vz_fit_b]) 
            vth_par_b_all[k] = v_th_par_fit_b
            vth_perp_b_all[k] = v_th_perp_fit_b
            kappa_all[k] = kappa_c
            fitted_params[k] = opt_params

        return (nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, fitted_params, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_all_1d, overlap_all_3d, goodness_all)

    if pick_model == 4:
        nc_all = np.zeros(N)
        vc_all = np.zeros([N, 3])
        vth_par_c_all = np.zeros(N)
        vth_perp_c_all = np.zeros(N)
        nb_all = np.full(N, np.nan)
        vb_all = np.full([N, 3], np.nan)
        vth_par_b_all = np.full(N, np.nan)
        vth_perp_b_all = np.full(N, np.nan)
        fitted_params = np.full((N, 12), np.nan)

        for k in range(len(results)):
            
            minimiser = results[k]
            
            opt_params = np.zeros(len(minimiser.params))
            for i, j in enumerate(minimiser.params):
                opt_params[i] = minimiser.params[j].value

            n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c = opt_params

            nc_all[k] = n_fit_c
            vc_all[k] = np.array([vx_fit_c, vy_fit_c, vz_fit_c]) 
            vth_par_c_all[k] = v_th_par_fit_c
            vth_perp_c_all[k] = v_th_perp_fit_c
            fitted_params[k, :6] = opt_params
        
        return (nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, fitted_params, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_all_1d, overlap_all_3d, goodness_all)
    
    if pick_model == 5:
        nc_all = np.zeros(N)
        vc_all = np.zeros([N, 3])
        vth_par_c_all = np.zeros(N)
        vth_perp_c_all = np.zeros(N)
        kappa_all = np.zeros(N)
        nb_all = np.full(N, np.nan)
        vb_all = np.full([N, 3], np.nan)
        vth_par_b_all = np.full(N, np.nan)
        vth_perp_b_all = np.full(N, np.nan)
        fitted_params = np.full((N, 12), np.nan)

        for k in range(len(results)):
            
            minimiser = results[k]
            
            opt_params = np.zeros(len(minimiser.params))
            for i, j in enumerate(minimiser.params):
                opt_params[i] = minimiser.params[j].value

            n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c = opt_params

            nc_all[k] = n_fit_c
            vc_all[k] = np.array([vx_fit_c, vy_fit_c, vz_fit_c]) 
            vth_par_c_all[k] = v_th_par_fit_c
            vth_perp_c_all[k] = v_th_perp_fit_c
            kappa_all[k] = kappa_c
            fitted_params[k, :6] = opt_params[:6]
        
        return (nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, fitted_params, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_all_1d, overlap_all_3d, goodness_all)
    
    if pick_model == 6:
        nc_all = np.zeros(N)
        vc_all = np.zeros([N, 3])
        vth_par_c_all = np.zeros(N)
        vth_perp_c_all = np.zeros(N)
        nb_all = np.zeros(N)
        vb_all = np.zeros([N, 3])
        vth_par_b_all = np.zeros(N)
        vth_perp_b_all = np.zeros(N)
        kappa_all = np.zeros(N)
        fitted_params = np.zeros((N, 12))

        for k in range(len(results)):
            
            minimiser = results[k]
            
            opt_params = np.zeros(len(minimiser.params))
            for i, j in enumerate(minimiser.params):
                opt_params[i] = minimiser.params[j].value

            n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b, kappa_b = opt_params
            vy_fit_b, vz_fit_b = vy_fit_c, vz_fit_c
            opt_params = n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b 

            nc_all[k] = n_fit_c
            vc_all[k] = np.array([vx_fit_c, vy_fit_c, vz_fit_c]) 
            vth_par_c_all[k] = v_th_par_fit_c
            vth_perp_c_all[k] = v_th_perp_fit_c
            nb_all[k] = n_fit_b
            vb_all[k] = np.array([vx_fit_b, vy_fit_b, vz_fit_b]) 
            vth_par_b_all[k] = v_th_par_fit_b
            vth_perp_b_all[k] = v_th_perp_fit_b
            kappa_all[k] = kappa_b
            fitted_params[k] = opt_params

        return (nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, kappa_all, fitted_params, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_all_1d, overlap_all_3d, goodness_all)

def load_hao_data(date1_str):

    # Load data from specific columns, skipping the first row
    use_cols = [7, 8, 10, 12, 13, 14, 15, 16, 17, 22, 23]
    path_hao = os.path.join('fit_results', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', 'results_hao.csv')
    data = np.loadtxt(path_hao, delimiter=',', skiprows=1, usecols=use_cols)
    nc_h, nb_h, n_h, vcpar_h, vcperp_h, vbpar_h, vbperp_h, vpar_h, vperp_h, Tpar_h, Tperp_h = data.T

    vc_h = np.linalg.norm(np.array([vcpar_h, vcperp_h]), axis=0)
    vb_h = np.linalg.norm(np.array([vbpar_h, vbperp_h]), axis=0)
    v_h = ((nc_h * vc_h) + (nb_h * vb_h)) / (n_h)

    T_h = (Tpar_h + Tperp_h + Tperp_h) / 3

    # print(data.T[:5])  # Show first 5 rows
    raw_dates = np.loadtxt(path_hao, delimiter=',', skiprows=1, usecols=[0], dtype=str)
    t_h = np.array([datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in raw_dates])  # adjust format if needed

    return (t_h, nc_h, vc_h, vpar_h, vperp_h, Tpar_h, Tperp_h, nb_h, vb_h, vbpar_h, vbperp_h, n_h, v_h, T_h)

def append_single_result_safe(
    ind_time,                 # datetime object
    opt_params_in,            # array shape (12,)
    param_names_in,           # list shape(12)
    in_conds,                 # array shape (12,)
    constraints_min,          # array shape (12,)
    constraints_max,          # array shape (12,)
    constraint_flag_all,      # array shape (12,)
    overlap_1d, overlap_3d,   # floats
    goodness_metric,           # float
    R_dist                    # float
):
    """
    Appends a single time instance's results to the shared HDF5 file.
    Safe for parallel use with a multiprocessing.Lock().
    """
    global H5_LOCK, H5_PATH

    # Ensure consistent parameter order
    param_names = param_names_in
    opt_params = np.asarray(opt_params_in, dtype=np.float64)[np.newaxis, :]

    # Prepare other arrays with leading axis = 1 for appending
    in_conds = np.asarray(in_conds, dtype=np.float64)[np.newaxis, :]     # (1, 12)
    constraints_all = np.stack([constraints_min, constraints_max, constraint_flag_all], axis=0)[np.newaxis, ...]  # (1, 3, 12)
    overlap_save = np.array([[overlap_1d, overlap_3d]], dtype=np.float64)  # (1, 2)
    goodness_arr = np.array([goodness_metric], dtype=np.float64)[np.newaxis, :]  # (1,)
    time_numeric = np.array([ind_time.timestamp()], dtype=np.float64)  # (1,)
    R_dist = np.array([R_dist], dtype=np.float64)[np.newaxis, :]  # (1,)

    with H5_LOCK:
        with h5py.File(H5_PATH, "a") as f:

            def append_ds(name, data):
                """Append data to dataset `name` in file `f`."""
                if name not in f:
                    maxshape = (None,) + data.shape[1:]
                    f.create_dataset(name, data=data, maxshape=maxshape,
                                     chunks=True, compression="gzip")
                else:
                    ds = f[name]
                    old = ds.shape[0]
                    ds.resize(old + data.shape[0], axis=0)
                    ds[old:] = data

            # Fit parameters: set param_names once
            if "fit_parameters" not in f:
                maxshape = (None,) + opt_params.shape[1:]
                dset = f.create_dataset("fit_parameters", data=opt_params,
                                        maxshape=maxshape, chunks=True, compression="gzip")
                dset.attrs["param_names"] = param_names
            else:
                append_ds("fit_parameters", opt_params)

            append_ds("initial_conditions", in_conds)
            append_ds("constraints", constraints_all)
            append_ds("overlap", overlap_save)
            append_ds("goodness_metric", goodness_arr)
            append_ds("time", time_numeric)
            append_ds("R", R_dist)

def fit_one_core_sep_par_drift_parallel_h5(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

    """
    
    ind, t_vdf, vx_bf, vy_bf, vz_bf, counts_in, vdf_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, method, nc_init, theta, vels, R_dist = tasks
    
    def clean_inputs(counts, G):
        counts_clean = np.copy(counts)
        G_clean = np.copy(G)
        invalid = ~np.isfinite(G_clean)
        G_clean[invalid] = np.nan
        counts_clean[invalid] = np.nan
        return counts_clean, G_clean

    dt = 1 / (9 * 96)  # PAS time acquisition per pixel

    if np.all((counts_in == 0) | np.isnan(counts_in)):

        append_single_result_safe(t_vdf, np.full(12, np.nan), param_names, np.full(12, np.nan),
                        np.full(12, np.nan), np.full(12, np.nan), np.full(12, np.nan),
                        np.nan, np.nan, np.nan, R_dist)
        return ind

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

        if np.linalg.norm(v_bulk_bf) < vels[ind_max]:
            core_mask = umag > np.linalg.norm(v_bulk_bf)
            vxb_init = -1.0

        elif np.linalg.norm(v_bulk_bf) > vels[ind_max]:    
            core_mask = umag < np.linalg.norm(v_bulk_bf)
            vxb_init = 1.0

        counts_core[~core_mask] = np.nan
    
    # if core data proccessing fails, return NaNs
    except Exception as e:
        print(f"Error processing counts_core at index {ind}: {e}")
        append_single_result_safe(t_vdf, np.full(12, np.nan), param_names, np.full(12, np.nan),
                                np.full(12, np.nan), np.full(12, np.nan), np.full(12, np.nan),
                                np.nan, np.nan, np.nan, R_dist)
        return ind
        
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
    
    constraints_max[0] = 1.5 * n_in
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
    
    n_beam = 0.18 * n_in
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
    
    n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, v_th_par_fit_b, v_th_perp_fit_b = opt_params

    opt_params_new = np.array([n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b])

    # Constraint flag - creates mask showing True when fitted params is equal to either the min or max constraint for each parameters
    constraint_flag_all = (constraints_max == opt_params_new) | (constraints_min == opt_params_new)

    "Compute overlap integral of two bi-Maxwellians, in 1d (integrate over angles and theta) and 3d"
    core_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
    beam_fit = bi_Max(vx_bf, vy_bf, vz_bf, n_fit_b, vx_fit_b, vy_fit_c, vz_fit_c, v_th_par_fit_b, v_th_perp_fit_b)

    min_fit = np.fmin(core_fit, beam_fit)
    overlap_3d = np.nansum(min_fit) / np.nansum(beam_fit)

    core_integrated = integrate_vdf_over_angles(core_fit, theta)
    beam_integrated = integrate_vdf_over_angles(beam_fit, theta)

    min_fit1d = np.fmin(core_integrated, beam_integrated)
    overlap_1d= np.nansum(min_fit1d) / np.nansum(beam_integrated)

    "Compute goodness of fit"
    vc = [vx_fit_c, vy_fit_c, vz_fit_c]
    vb = [vx_fit_b, vy_fit_c, vz_fit_c]
    goodness_metric = goodness_of_fit_one(n_fit_c, vc, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b, vb, v_th_par_fit_b, v_th_perp_fit_b, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt)

    param_names = ['n_c', 'vx_c', 'vy_c', 'vz_c', 'vth_par_c', 'vth_perp_c', 'n_b', 'vx_b', 'vy_b', 'vz_b', 'vth_par_b', 'vth_perp_b']

    append_single_result_safe(t_vdf, opt_params_new, param_names, in_conds,
                                constraints_min, constraints_max, constraint_flag_all,
                                overlap_1d, overlap_3d, goodness_metric, R_dist)

    return ind
    
def fit_one_core_sep_par_drift_parallel_h5_new(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

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

def fit_one_core_only_parallel_h5_new(tasks):
    """
    Fitting function 

    Parameters
    ----------
    ind : datetime
        time instance.
    args : tuple
        arguments for fit:
        - Parameter object
        - ux
        - uy
        - uz
        - vdf data
        - count data
        - n
        - v_bulk_bf
        - T_par
        - T_perp
        - T
        - G factor
        - B_all

    Returns
    -------
    None.

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


def init_worker_h5(lock, h5_path):
    """Initializer called once in each worker process."""
    global H5_LOCK, H5_PATH
    H5_LOCK = lock
    H5_PATH = h5_path

def run_parallel_h5(tasks, fun, h5_path, n_workers=32):
    # Create a multiprocessing lock and start pool with initializer
    lock = mp.Lock()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker_h5, initargs=(lock, h5_path)) as exe:
        futures = [exe.submit(fun, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fitting progress"):
            try:
                ind = fut.result()
            except Exception as exc:
                print("Worker error:", exc)

def save_all_results_h5_old(
    h5_path,
    times,          # (N, 1) float64 (timestamps)
    opt_params,     # (N, P) float64
    param_names,    # list of P strings
    in_conds,       # (N, P) float64
    constraints,    # (N, P) float64
    overlap,        # (N, 1) float64
    ks_metric,      # (N, 2) float64
    R,              # (N, 1) float64
    n_mom,          # (N, 1) float64
    v_mom,          # (N, 3) float64
    T_mom,          # (N, 2) float64
    B               # (N, 3) float63
):
    """
    Append all results to an existing HDF5 file (or create if not exists).
    Deduplicates by timestamp to avoid overlaps between multiple runs.
    """

    times = np.asarray(times, dtype=np.float64).reshape(-1, 1)
    opt_params = np.asarray(opt_params, dtype=np.float64)
    in_conds = np.asarray(in_conds, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    ks_metric = np.asarray(ks_metric, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n_mom = np.asarray(n_mom, dtype=np.float64)
    v_mom = np.asarray(v_mom, dtype=np.float64)
    T_mom = np.asarray(T_mom, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

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
        append_ds("time", times, dedup=True)
        append_ds("R", R)
        append_ds('n_mom', n_mom)
        append_ds('v_mom', v_mom)
        append_ds('T_mom', T_mom)
        append_ds('B', B)

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

def run_parallel_h5_new(tasks, fun, h5_path, n_workers=32, core_only=False):
    """
    Run tasks in parallel, collect results, and save them into HDF5
    at the end using `save_all_results_h5`. No duplicates per interval,
    and safe across multiple runs (appends to file if it exists).
    """

    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = [exe.submit(fun, task) for task in tasks]

        for ind_fit, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Fitting progress")):
            try:
                res = fut.result()
                results.append(res)

            except Exception as exc:
                print("Worker error1:", exc)
                task_temp = tasks[ind_fit]
                # Create dummy result with NaNs
                t_nan = task_temp[1] # fallback timestamp
                R_nan = task_temp[-1]  # fallback R_dist
                if core_only == True:
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
                    param_names_nan = ['n_c', 'vx_c', 'vy_c', 'vz_c', 'vth_par_c', 'vth_perp_c', 'n_b', 'vx_b', 'vy_b', 'vz_b', 'vth_par_b', 'vth_perp_b']
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

def run_serial_h5_new(tasks, fun, h5_path, core_only=False):
    """
    Run tasks serially, collect results, and save them into HDF5
    at the end using `save_all_results_h5`. No duplicates per interval,
    and safe across multiple runs (appends to file if it exists).
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

def save_results_h5_single(
    ind_time,                 # datetime object
    opt_params_in,            # array shape (12,)
    param_names_in,           # list shape (12)
    in_conds,                 # array shape (12,)
    constraints_min,          # array shape (12,)
    constraints_max,          # array shape (12,)
    constraint_flag_all,      # array shape (12,)
    overlap_1d, overlap_3d,   # floats
    goodness_metric,          # float
    R_dist,                   # float
    h5_path,                  # str: path to HDF5 file
    lock                      # multiprocessing.Lock
):
    """Appends a single result safely into the shared HDF5 file."""

    opt_params = np.asarray(opt_params_in, dtype=np.float64)[np.newaxis, :]
    in_conds = np.asarray(in_conds, dtype=np.float64)[np.newaxis, :]
    constraints_all = np.stack([constraints_min, constraints_max, constraint_flag_all],
                               axis=0)[np.newaxis, ...]
    overlap_save = np.array([[overlap_1d, overlap_3d]], dtype=np.float64)
    goodness_arr = np.array([goodness_metric], dtype=np.float64)[np.newaxis, :]
    time_numeric = np.array([ind_time.timestamp()], dtype=np.float64)
    R_dist = np.array([R_dist], dtype=np.float64)[np.newaxis, :]

    with lock:  # only one process writes at a time
        with h5py.File(h5_path, "a") as f:

            def append_ds2(name, data):
                """Append data to dataset `name` in file `f`."""
                if name not in f:
                    maxshape = (None,) + data.shape[1:]
                    f.create_dataset(name, data=data, maxshape=maxshape,
                                     chunks=True, compression="gzip")
                else:
                    ds = f[name]
                    old = ds.shape[0]
                    ds.resize(old + data.shape[0], axis=0)
                    ds[old:] = data

            # Fit parameters: store param_names as attribute only once
            if "fit_parameters" not in f:
                dset = f.create_dataset("fit_parameters", data=opt_params,
                                        maxshape=(None, opt_params.shape[1]),
                                        chunks=True, compression="gzip")
                dset.attrs["param_names"] = param_names_in
            else:
                append_ds2("fit_parameters", opt_params)

            append_ds2("initial_conditions", in_conds)
            append_ds2("constraints", constraints_all)
            append_ds2("overlap", overlap_save)
            append_ds2("goodness_metric", goodness_arr)
            append_ds2("time", time_numeric)
            append_ds2("R", R_dist)
                
def fit_data_h5(pick_model, t_vdf, vdf_in, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, vels, theta, R_dist, method_in = 'powell', file_name = f'results_powell_cbc_par_drift_TEST'):
    """
    CHOOSE THE FITTING MODEL, FIT AND SAVE DATA!
    pick_model determines which fitting model to use.
    --------------------------------------------------------------------------------------------------------------------
    pick_model = 0 - Separate core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 1 - Core and beam fit together, only parallel drift, pick beam direction automatically!
    pick_model = 2 - Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 3 - Kappa core and beam fit together, only parallel drift, pick beam direction automatically!
    pick model = 4 - Core only Bi-Maxwellian fit!
    pick model = 5 - Core only Bi-kappa fit!
    pick_model = 6 - Separate bi_Max core and Kappa beam fit, only parallel drift, pick beam direction automatically!
     --------------------------------------------------------------------------------------------------------------------
    """

    N = len(vdf_in)
    theta_copies = [theta.copy() for _ in range(N)]

    if pick_model == 0:
        "*********************************************************************"
        "Separate core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_core_sep_par_drift_parallel_h5
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = file_name
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i], R_dist[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 1:
        "*********************************************************************"
        "Core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_both_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_par_drift'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"
        
    if pick_model == 2:
        "*********************************************************************"
        "Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_sep_kappa_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_cbc_kappa'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 3:
        "*********************************************************************"
        "Kappa core and beam fit together, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_both_kappa_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_kappa'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, t_vdf[i],vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"
        
    if pick_model == 4:
        "*********************************************************************"
        "Core only Bi-Maxwellian fit!"
        "********************************************************************"
        fit_in = fit_one_core_only
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        save_path = f'results_{method[0]}_core_only'
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i]) for i in range(N)]
        "********************************************************************"

    if pick_model == 5:
        "*********************************************************************"
        "Core only Kappa fit!"
        "********************************************************************"
        fit_in = fit_one_core_only_kappa
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        save_path = f'results_{method[0]}_core_only_kappa'
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i]) for i in range(N)]
        "********************************************************************"
    
    if pick_model == 6:
        "*********************************************************************"
        "Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_sep_kappa_beam_par_drift_parallel
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = f'results_{method[0]}_cbc_kappa_beam'
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i]) for i in range(N)]
        "********************************************************************"

    run_parallel_h5(tasks_in, fit_in, n_workers=32, h5_path=save_path)

def fit_data_h5_new(pick_model, t_vdf, vdf_in, vx_bf, vy_bf, vz_bf, counts_in, n, v_bulk_bf, T_par, T_perp, T, G_factors, B_all, vels, theta, R_dist, qf, method_in = 'powell', file_name = f'results_powell_cbc_par_drift_TEST', n_workers=32):
    """
    CHOOSE THE FITTING MODEL, FIT AND SAVE DATA!
    pick_model determines which fitting model to use.
    --------------------------------------------------------------------------------------------------------------------
    pick_model = 0 - Separate core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 1 - Core and beam fit together, only parallel drift, pick beam direction automatically!
    pick_model = 2 - Separate Kappa core and beam fit, only parallel drift, pick beam direction automatically!
    pick_model = 3 - Kappa core and beam fit together, only parallel drift, pick beam direction automatically!
    pick model = 4 - Core only Bi-Maxwellian fit!
    pick model = 5 - Core only Bi-kappa fit!
    pick_model = 6 - Separate bi_Max core and Kappa beam fit, only parallel drift, pick beam direction automatically!
     --------------------------------------------------------------------------------------------------------------------
    """

    N = len(vdf_in)
    theta_copies = [theta.copy() for _ in range(N)]

    if pick_model == 0:
        "*********************************************************************"
        "Separate core and beam fit, only parallel drift, pick beam direction automatically!"
        "********************************************************************"
        fit_in = fit_one_core_sep_par_drift_parallel_h5_new
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
        fit_in = fit_one_core_only_parallel_h5_new
        method = [method_in] * N  # 'powell' or 'differential_evolution'
        nc_init = np.ones(N) * 1.1
        save_path = file_name
        vels_copies = [vels.copy()*1e3 for _ in range(N)]
        core_only = True
        tasks_in = [(i, t_vdf[i], vx_bf[i], vy_bf[i], vz_bf[i], counts_in[i], vdf_in[i], n[i], v_bulk_bf[i], T_par[i], T_perp[i], T[i], G_factors[i], B_all[i], method[i], nc_init[i], theta_copies[i], vels_copies[i], R_dist[i], qf[i]) for i in range(N)]        
        "********************************************************************"
        
    run_serial_h5_new(tasks_in, fit_in, h5_path=save_path, core_only=core_only)

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
    t_vdf = data['time']
    sorted_indices = np.argsort(t_vdf)
    t_vdf = t_vdf[sorted_indices]
    data_transposed = np.array([col for col in data['fit_parameters'].T])[:, sorted_indices]
    nc_all, vxc, vyc, vzc, vth_par_c_all, vth_perp_c_all, nb_all, vxb, vyb, vzb, vth_par_b_all, vth_perp_b_all = data_transposed
    vc_all = np.stack((vxc, vyc, vzc), axis = 1)
    vb_all = np.stack((vxb, vyb, vzb), axis = 1)
    in_conds = data['initial_conditions'][sorted_indices]
    constraints_min, constraints_max, constraint_flag_all = data['constraints'][:, 0][sorted_indices], data['constraints'][:, 1][sorted_indices], data['constraints'][:, 2][sorted_indices]
    overlap_all_1d, overlap_all_3d = data['overlap'][:, 0][sorted_indices], data['overlap'][:, 1][sorted_indices]
    goodness_all = data['goodness_metric'][sorted_indices]
    R_all = data['R'][sorted_indices]
    fitted_params = data_transposed.T

    return nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, fitted_params, in_conds, constraints_min, constraints_max, constraint_flag_all, overlap_all_1d, overlap_all_3d, goodness_all, t_vdf, R_all

def format_data_h5_new(data):
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

def format_data_h5_core_new(data):
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
