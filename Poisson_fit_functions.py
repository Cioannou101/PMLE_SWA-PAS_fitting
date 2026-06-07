"""
Author: Charalambos Ioannou
Institution: UCL / MSSL
Email: charalambos.ioannou.22@ucl.ac.uk
GitHub: @Cioannou101
Created: 2026-06-07

This script contains functions used in performing Poisson likelihood fits to the VDF data.
"""

import numpy as np
import scipy.constants as sc
import lmfit

def log_fact_approx(n):
    "Srinivasa Ramanujan formula for log(n!) approximation - https://en.wikipedia.org/wiki/Stirling%27s_approximation"
    if n == 0:
        return 0
    else:
        return (n*np.log(n)) - n + (np.log((n*(1+4*n*(1+2*n))) + (1/30)) / 6) + (np.log(np.pi)/2)
    
def log_fact_approx2(n):
    """
    Srinivasa Ramanujan formula for log(n!) approximation,
    now vectorized to work with numpy arrays.
    """
    n = np.asarray(n)  # Ensure n is an array
    result = np.zeros_like(n, dtype=float)

    # Where n == 0, log(0!) = 0
    mask_zero = (n == 0)
    result[mask_zero] = 0

    # Where n > 0, apply the formula
    mask_pos = ~mask_zero
    n_pos = n[mask_pos]
    result[mask_pos] = (
        (n_pos * np.log(n_pos))
        - n_pos
        + (np.log((n_pos * (1 + 4 * n_pos * (1 + 2 * n_pos))) + (1/30)) / 6)
        + (np.log(np.pi) / 2)
    )

    return result

def log_fact_term(n):
    
    n = round(n) # ensure n is an integer
    fact_term = 0
    for i in range(1, n + 1):
        fact_term += np.log(i)   
    
    return fact_term

def poisson_log_likelihood2(data, model):
    
    x = data.flatten()
    lam = model.flatten()

    # Avoid issues with zeros
    # lam = np.clip(lam, 1e-10, None)

    fact_terms = log_fact_approx2(x)
    log_likelihood_array = -fact_terms - lam + (x * np.log(lam))

    return np.sum(log_likelihood_array)

def poisson_log_likelihood3(data, model):

    x = data.flatten()
    lam = model.flatten()

    zero_mask = lam == 0
    x = x[~zero_mask]
    lam = lam[~zero_mask]
    # Avoid issues with zeros
    # lam = np.clip(lam, 1e-10, None)

    fact_terms = log_fact_approx2(x)
    log_likelihood_array = -fact_terms - lam + (x * np.log(lam))

    return np.sum(log_likelihood_array)

def poisson_log_likelihood4(data, model):
    
    x = data.flatten()
    lam = model.flatten()

    # Avoid issues with zeros
    # lam = np.clip(lam, 1e-10, None)

    fact_terms = log_fact_approx2(x)
    log_likelihood_array_signed = np.sign(lam-x)*(-fact_terms - lam + (x * np.log(lam)))

    return - np.sum(abs(log_likelihood_array_signed)), log_likelihood_array_signed

def counts_biMaxwellian(ux, uy, uz, n_var, v_x, v_y, v_z, v_th_par, v_th_perp, G, dt):
    """
    Produces counts following a bi-Maxwellian VDF.

    Parameters
    ----------
    ux : array
        Velocity in the x (parallel) direction.
    uy : array
        Velocity in the y direction.
    uz : array
        Velocity in the z direction.
    n_var : float
        The plasma density in m^{-3}.
    v_x : array
        Bulk velocity in the x (parallel) direction in m/s.
    v_y : array
        Bulk velocity in the y direction in m/s.
    v_z : array
        Bulk velocity in the z direction in m/s.
    v_th_par : float
        Thermal velocity in paraller (x) direction in m/s.
    v_th_perp : float
        Thermal velocity in perpendicular direction in m/s.
    G   : float
        Geometric factor
    dt  : float
        Acquisition time in s

    Returns
    -------
    counts : array
        The counts distribution.

    """
    vel_mag = np.linalg.norm(np.array([ux, uy, uz]), axis=0)
    E = 0.5 * sc.m_p * vel_mag * vel_mag
    vel_par = ux - v_x

    vy_perp = uy - v_y
    vz_perp = uz - v_z
    vel_perp = np.sqrt((vy_perp * vy_perp) + (vz_perp * vz_perp))

    denominator = (np.pi ** 1.5) * v_th_par * v_th_perp * v_th_perp
    term1 = n_var / denominator
    exponent = ((vel_par * vel_par) / (v_th_par * v_th_par)) + \
        ((vel_perp * vel_perp) / (v_th_perp * v_th_perp))

    vdf = term1 * np.exp(- exponent)

    counts = 2 * G * dt * (E**2) * vdf / (sc.m_p**2)

    return counts

def counts_double_biMaxwellian(ux, uy, uz, n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1, n_var2, v_x2, v_y2, v_z2, v_th_par2, v_th_perp2, G, dt):

    # vel_mag = np.linalg.norm(np.array([ux, uy, uz]), axis=0)
    # E = 0.5 * sc.m_p * vel_mag * vel_mag
    vel_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    E2 = (0.5 * sc.m_p * vel_mag**2)**2

    # CORE
    vel_par1 = ux - v_x1

    vy_perp1 = uy - v_y1
    vz_perp1 = uz - v_z1
    vel_perp1 = np.sqrt((vy_perp1 * vy_perp1) + (vz_perp1 * vz_perp1))

    denominator1 = (np.pi ** 1.5) * v_th_par1 * v_th_perp1 * v_th_perp1
    term1 = n_var1 / denominator1
    exponent1 = ((vel_par1 * vel_par1) / (v_th_par1 * v_th_par1)) + \
        ((vel_perp1 * vel_perp1) / (v_th_perp1 * v_th_perp1))

    vdf1 = term1 * np.exp(- exponent1)

    counts1 = 2 * G * dt * (E2) * vdf1 / (sc.m_p**2)

    # BEAM
    vel_par2 = ux - v_x2

    vy_perp2 = uy - v_y2
    vz_perp2 = uz - v_z2
    vel_perp2 = np.sqrt((vy_perp2 * vy_perp2) + (vz_perp2 * vz_perp2))

    denominator2 = (np.pi ** 1.5) * v_th_par2 * v_th_perp2 * v_th_perp2
    term2 = n_var2 / denominator2
    exponent2 = ((vel_par2 * vel_par2) / (v_th_par2 * v_th_par2)) + \
        ((vel_perp2 * vel_perp2) / (v_th_perp2 * v_th_perp2))

    vdf2 = term2 * np.exp(- exponent2)

    counts2 = 2 * G * dt * (E2) * vdf2 / (sc.m_p**2)
    
    # # Set nans to zeros
    # mask = counts1 

    return counts1 + counts2

def logp_minimisation(params, vx, vy, vz, G, dt, data):

    vals = params.valuesdict()
    n_in = vals['n_var']
    vx_bulk_in = vals['vx']
    vy_bulk_in = vals['vy']
    vz_bulk_in = vals['vz']
    v_th_par_in = vals['v_th_par']
    v_th_perp_in = vals['v_th_perp']

    model = counts_biMaxwellian(vx, vy, vz, n_in, vx_bulk_in, vy_bulk_in,
                                vz_bulk_in, v_th_par_in, v_th_perp_in, G, dt)

    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]

    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)

    # return negative of log likelihood
    return - log_likelihood

def logp_double_minimisation(params, vx, vy, vz, G, dt, data):
    
    vals = params.valuesdict()
    
    # CORE parameters
    n1, vx1, vy1, vz1, vth_par1, vth_perp1 = (
        vals['n_var1'], vals['vx1'], vals['vy1'], vals['vz1'],
        vals['v_th_par1'], vals['v_th_perp1']
    )
    
    # BEAM parameters
    n2, vx2, vy2, vz2, vth_par2, vth_perp2 = (
        vals['n_var2'], vals['vx2'], vals['vy2'], vals['vz2'],
        vals['v_th_par2'], vals['v_th_perp2']
    )

    model = counts_double_biMaxwellian(
     vx, vy, vz,
     n1, vx1, vy1, vz1, vth_par1, vth_perp1,
     n2, vx2, vy2, vz2, vth_par2, vth_perp2,
     G, dt
 )
    
    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]
    
    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)

    # return negative of log likelihood
    return - log_likelihood

def double_bi_Max(ux, uy, uz, n_var1, vx1, vy1, vz1, v_th_par1, v_th_perp1, n_var2, vx2, vy2, vz2, v_th_par2, v_th_perp2):

    vel_par1 = ux - vx1

    vy_perp1 = uy - vy1
    vz_perp1 = uz - vz1
    vel_perp1 = np.sqrt((vy_perp1 * vy_perp1) + (vz_perp1 * vz_perp1))

    denominator1 = (np.pi ** 1.5) * v_th_par1 * v_th_perp1 * v_th_perp1
    term1 = n_var1 / denominator1
    exponent1 = ((vel_par1 * vel_par1) / (v_th_par1 * v_th_par1)) + \
        ((vel_perp1 * vel_perp1) / (v_th_perp1 * v_th_perp1))

    f1 = term1 * np.exp(- exponent1)

    vel_par2 = ux - vx2

    vy_perp2 = uy - vy2
    vz_perp2 = uz - vz2
    vel_perp2 = np.sqrt((vy_perp2 * vy_perp2) + (vz_perp2 * vz_perp2))

    denominator2 = (np.pi ** 1.5) * v_th_par2 * v_th_perp2 * v_th_perp2
    term2 = n_var2 / denominator2
    exponent2 = ((vel_par2 * vel_par2) / (v_th_par2 * v_th_par2)) + \
        ((vel_perp2 * vel_perp2) / (v_th_perp2 * v_th_perp2))

    f2 = term2 * np.exp(- exponent2)

    # if np.isfinite(f).any() == False:
    #    return 0

    return f1 + f2

def bi_Max(ux, uy, uz, n_var, v_x, v_y, v_z, v_th_par, v_th_perp):
    """
    Produces a VDF following a bi-Maxwellian distribution.

    Parameters
    ----------
    ux : array
        Velocity in the x (parallel) direction.
    uy : array
        Velocity in the y direction.
    uz : array
        Velocity in the z direction.
    n_var : float
        The plasma density in m^{m-3}.
    v_x : array
        Bulk velocity in the x (parallel) direction.
    v_y : array
        Bulk velocity in the y direction.
    v_z : array
        Bulk velocity in the z direction.
    v_th_par : float
        Thermal velocity in paraller (x) direction.
    v_th_perp : float
        Thermal velocity in perpendicular direction.

    Returns
    -------
    f : array
        The Velocity distribution function VDF.

    """

    vel_par = ux - v_x

    vy_perp = uy - v_y
    vz_perp = uz - v_z
    vel_perp = np.sqrt((vy_perp * vy_perp) + (vz_perp * vz_perp))

    denominator = (np.pi ** 1.5) * v_th_par * v_th_perp * v_th_perp
    term1 = n_var / denominator
    exponent = ((vel_par * vel_par) / (v_th_par * v_th_par)) + \
        ((vel_perp * vel_perp) / (v_th_perp * v_th_perp))

    f = term1 * np.exp(- exponent)

    # if np.isfinite(f).any() == False:
    #    return 0

    return f

def goodness_of_fit_one(nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt):
    """    
    Calculate the goodness of fit for each set of parameters.
    """
            
    n_fit_c = nc_all
    vx_fit_c = vc_all[0]
    vy_fit_c = vc_all[1]  
    vz_fit_c = vc_all[2]  
    v_th_par_fit_c = vth_par_c_all
    v_th_perp_fit_c = vth_perp_c_all
    
    n_fit_b = nb_all
    vx_fit_b = vb_all[0]
    vy_fit_b = vb_all[1]  
    vz_fit_b = vb_all[2]  
    v_th_par_fit_b = vth_par_b_all
    v_th_perp_fit_b = vth_perp_b_all

    fit_params = lmfit.Parameters()
    fit_params.add('n_var1', value=n_fit_c)
    fit_params.add('vx1', value=vx_fit_c)
    fit_params.add('vy1', value=vy_fit_c)
    fit_params.add('vz1', value=vz_fit_c)
    fit_params.add('v_th_par1', value=v_th_par_fit_c)
    fit_params.add('v_th_perp1', value=v_th_perp_fit_c)
    
    fit_params.add('n_var2', value=n_fit_b)
    fit_params.add('vx2', value=vx_fit_b)
    fit_params.add('vy2', value=vy_fit_b)
    fit_params.add('vz2', value=vz_fit_b)
    fit_params.add('v_th_par2', value=v_th_par_fit_b)
    fit_params.add('v_th_perp2', value=v_th_perp_fit_b)

    counts_data_test = counts_in.copy()
    dt = 1 / (9 * 96)
    G_in = np.copy(G_factors)
    
    mask_data = ~np.isfinite(G_in)
    G_in[mask_data] = np.nan
    counts_data_test[mask_data] = np.nan
    
    counts_fit = counts_double_biMaxwellian(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b,
                            vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b, G_in, dt)
    
    probs_data = logp_double_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_data_test)
    probs_model = logp_double_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_fit)

    # mean_ratio = np.sum(probs_data / probs_model) / (len(probs_data))
    mean_ratio =  probs_model / probs_data
    
    return mean_ratio

def logp_double_minimisation_par_drift(params, vx, vy, vz, G, dt, data):
    
    vals = params.valuesdict()
    
    # CORE parameters
    n1, vx1, vy1, vz1, vth_par1, vth_perp1 = (
        vals['n_var1'], vals['vx1'], vals['vy'], vals['vz'],
        vals['v_th_par1'], vals['v_th_perp1']
    )
    
    # BEAM parameters
    n2, vx2, vy2, vz2, vth_par2, vth_perp2 = (
        vals['n_var2'], vals['vx2'], vals['vy'], vals['vz'],
        vals['v_th_par2'], vals['v_th_perp2']
    )

    model = counts_double_biMaxwellian(
     vx, vy, vz,
     n1, vx1, vy1, vz1, vth_par1, vth_perp1,
     n2, vx2, vy2, vz2, vth_par2, vth_perp2,
     G, dt
 )
    
    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]
    
    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)

    # return negative of log likelihood
    return - log_likelihood

def goodness_of_fit_one_core_only(nc_all, vc_all, vth_par_c_all, vth_perp_c_all, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt):
    """    
    Calculate the goodness of fit for each set of parameters.
    """
            
    n_fit_c = nc_all
    vx_fit_c = vc_all[0]
    vy_fit_c = vc_all[1]  
    vz_fit_c = vc_all[2]  
    v_th_par_fit_c = vth_par_c_all
    v_th_perp_fit_c = vth_perp_c_all

    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_fit_c)
    fit_params.add('vx', value=vx_fit_c)
    fit_params.add('vy', value=vy_fit_c)
    fit_params.add('vz', value=vz_fit_c)
    fit_params.add('v_th_par', value=v_th_par_fit_c)
    fit_params.add('v_th_perp', value=v_th_perp_fit_c)

    counts_data_test = counts_in.copy()
    dt = 1 / (9 * 96)
    G_in = np.copy(G_factors)
    
    mask_data = ~np.isfinite(G_in)
    G_in[mask_data] = np.nan
    counts_data_test[mask_data] = np.nan
    
    counts_fit = counts_biMaxwellian(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, G_in, dt)
    
    probs_data = logp_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_data_test)
    probs_model = logp_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_fit)

    # mean_ratio = np.sum(probs_data / probs_model) / (len(probs_data))
    mean_ratio =  probs_model / probs_data
    
    return mean_ratio
