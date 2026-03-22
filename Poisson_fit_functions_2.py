# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:19:43 2025

@author: CI
"""

import numpy as np
import scipy.constants as sc
from scipy.special import factorial
from scipy.special import gamma
import lmfit
from tqdm import tqdm

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
    
def log_Poisson_prob(counts, mean):
    
    fact_term = log_fact_term(counts)   
    
    log_prob = - fact_term - mean + (counts * np.log(mean))
    
    return log_prob

def poisson_log_likelihood(data, model):

    x = data.flatten()
    lam = model.flatten()
    
    log_likelihood = 0
    for i in range(len(x)):
        
        # print(x[i])
        # np.log(lam[i])
        # print(lam[i], np.log(lam[i]))
        fact_term = log_fact_approx(x[i])
        # fact_term = log_fact_term(x[i])
        
        log_likelihood += - fact_term - lam[i] + (x[i] * np.log(lam[i]))
        
        # if x[i] < 20:
        #     log_likelihood += - np.log(factorial(x[i])) - lam[i] + (x[i] * np.log(lam[i]))

        # else:
        #     log_likelihood += - log_fact_approx(x[i]) - lam[i] + (x[i] * np.log(lam[i]))

    return log_likelihood

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
    Produces a counts following a bi-Maxwellian VDF.

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

def counts_triple_biMaxwellian(ux, uy, uz, n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1, n_var2, v_x2, v_y2, v_z2, v_th_par2, v_th_perp2, n_var3, v_x3, v_y3, v_z3, v_th_par3, v_th_perp3, G, dt):

    vel_mag = np.linalg.norm(np.array([ux, uy, uz]), axis=0)
    E = 0.5 * sc.m_p * vel_mag * vel_mag

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

    counts1 = 2 * G * dt * (E**2) * vdf1 / (sc.m_p**2)

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

    counts2 = 2 * G * dt * (E**2) * vdf2 / (sc.m_p**2)

    # ALPHA
    vel_par3 = ux - v_x3

    vy_perp3 = uy - v_y3
    vz_perp3 = uz - v_z3
    vel_perp3 = np.sqrt((vy_perp3 * vy_perp3) + (vz_perp3 * vz_perp3))

    denominator3 = (np.pi ** 1.5) * v_th_par3 * v_th_perp3 * v_th_perp3
    term3 = n_var3 / denominator3
    exponent3 = ((vel_par3 * vel_par3) / (v_th_par3 * v_th_par3)) + \
        ((vel_perp3 * vel_perp3) / (v_th_perp3 * v_th_perp3))

    vdf3 = term3 * np.exp(- exponent3)

    counts3 = 2 * G * dt * (E**2) * vdf3 / (sc.m_p**2)
    
    # # Set nans to zeros
    # mask = counts1 

    return counts1 + counts2 + counts3

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

def logp_triple_minimisation(params, vx, vy, vz, G, dt, data):

    vals = params.valuesdict()
    # CORE
    n_in1 = vals['n_var1']
    vx_bulk_in1 = vals['vx1']
    vy_bulk_in1 = vals['vy1']
    vz_bulk_in1 = vals['vz1']
    v_th_par_in1 = vals['v_th_par1']
    v_th_perp_in1 = vals['v_th_perp1']

    # BEAM
    n_in2 = vals['n_var2']
    vx_bulk_in2 = vals['vx2']
    vy_bulk_in2 = vals['vy2']
    vz_bulk_in2 = vals['vz2']
    v_th_par_in2 = vals['v_th_par2']
    v_th_perp_in2 = vals['v_th_perp2']

    # ALPHA
    n_in3 = vals['n_var3']
    vx_bulk_in3 = vals['vx3']
    vy_bulk_in3 = vals['vy3']
    vz_bulk_in3 = vals['vz3']
    v_th_par_in3 = vals['v_th_par3']
    v_th_perp_in3 = vals['v_th_perp3']

    model = counts_triple_biMaxwellian(vx, vy, vz, n_in1, vx_bulk_in1, vy_bulk_in1, vz_bulk_in1, v_th_par_in1, v_th_perp_in1, n_in2, vx_bulk_in2,
                                       vy_bulk_in2, vz_bulk_in2, v_th_par_in2, v_th_perp_in2, n_in3, vx_bulk_in3, vy_bulk_in3, vz_bulk_in3, v_th_par_in3, v_th_perp_in3, G, dt)

    mask = np.isfinite(data)
    # mask = model > 1e-50
    data = data[mask]
    model = model[mask]
    # print(data.shape)
    # print(model.shape)
    
    # mask = model != 0
    mask = model > 1e-50
    data = data[mask]
    model = model[mask]
    # print(model.shape)
    
    # mask = np.isfinite(model)
    # print(np.isfinite(model).all())
    # data = data[mask]
    # model = model[mask]

    log_likelihood = poisson_log_likelihood(data, model)

    # return negative of log likelihood
    return - log_likelihood

def triple_bi_Max(ux, uy, uz, n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1, n_var2, v_x2, v_y2, v_z2, v_th_par2, v_th_perp2, n_var3, v_x3, v_y3, v_z3, v_th_par3, v_th_perp3):

    vel_par1 = ux - v_x1

    vy_perp1 = uy - v_y1
    vz_perp1 = uz - v_z1
    vel_perp1 = np.sqrt((vy_perp1 * vy_perp1) + (vz_perp1 * vz_perp1))

    denominator1 = (np.pi ** 1.5) * v_th_par1 * v_th_perp1 * v_th_perp1
    term1 = n_var1 / denominator1
    exponent1 = ((vel_par1 * vel_par1) / (v_th_par1 * v_th_par1)) + \
        ((vel_perp1 * vel_perp1) / (v_th_perp1 * v_th_perp1))

    f1 = term1 * np.exp(- exponent1)

    vel_par2 = ux - v_x2

    vy_perp2 = uy - v_y2
    vz_perp2 = uz - v_z2
    vel_perp2 = np.sqrt((vy_perp2 * vy_perp2) + (vz_perp2 * vz_perp2))

    denominator2 = (np.pi ** 1.5) * v_th_par2 * v_th_perp2 * v_th_perp2
    term2 = n_var2 / denominator2
    exponent2 = ((vel_par2 * vel_par2) / (v_th_par2 * v_th_par2)) + \
        ((vel_perp2 * vel_perp2) / (v_th_perp2 * v_th_perp2))

    f2 = term2 * np.exp(- exponent2)

    vel_par3 = ux - v_x3

    vy_perp3 = uy - v_y3
    vz_perp3 = uz - v_z3
    vel_perp3 = np.sqrt((vy_perp3 * vy_perp3) + (vz_perp3 * vz_perp3))

    denominator3 = (np.pi ** 1.5) * v_th_par3 * v_th_perp3 * v_th_perp3
    term3 = n_var3 / denominator3
    exponent3 = ((vel_par3 * vel_par3) / (v_th_par3 * v_th_par3)) + \
        ((vel_perp3 * vel_perp3) / (v_th_perp3 * v_th_perp3))

    f3 = term3 * np.exp(- exponent3)

    # if np.isfinite(f).any() == False:
    #    return 0

    return f1 + f2 + f3

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

def bi_kappa(vx, vy, vz, n, vx0, vy0, vz0, vth_par, vth_perp, kappa):
    """
    Computes a bi-Kappa distribution in velocity space with parallel direction along x.

    Parameters:
    - vx, vy, vz: velocity arrays (broadcastable)
    - n: density [m^-3]
    - vx0, vy0, vz0: drift velocities [m/s]
    - vth_par: parallel thermal speed [m/s]
    - vth_perp: perpendicular thermal speed [m/s]
    - kappa: kappa index (must be > 3/2)

    Returns:
    - f: bi-Kappa distribution values
    """

    # Shifted velocity components
    dvx = vx - vx0  # parallel direction
    dvy = vy - vy0
    dvz = vz - vz0

    # Squared velocity terms
    v_par_sq = dvx**2 / vth_par**2
    v_perp_sq = dvy**2 / vth_perp**2 + dvz**2 / vth_perp**2

    # Kappa argument
    A_kappa = 1 + (v_par_sq + v_perp_sq) / (kappa - 1.5)

    # Normalization constant
    norm = (
        n * gamma(kappa + 1) /
        (gamma(kappa - 0.5) * (np.pi * (kappa - 1.5))**1.5 *
         vth_par * vth_perp**2)
    )

    f = norm * A_kappa**(-kappa - 1)

    return f

def bi_kappa_counts(vx, vy, vz, n, vx0, vy0, vz0, vth_par, vth_perp, kappa, G, dt):
    """
    Computes a bi-Kappa distribution in velocity space with parallel direction along x.

    Parameters:
    - vx, vy, vz: velocity arrays (broadcastable)
    - n: density [m^-3]
    - vx0, vy0, vz0: drift velocities [m/s]
    - vth_par: parallel thermal speed [m/s]
    - vth_perp: perpendicular thermal speed [m/s]
    - kappa: kappa index (must be > 3/2)

    Returns:
    - f: bi-Kappa distribution values
    """

    # Shifted velocity components
    dvx = vx - vx0  # parallel direction
    dvy = vy - vy0
    dvz = vz - vz0

    # Squared velocity terms
    v_par_sq = dvx**2 / vth_par**2
    v_perp_sq = dvy**2 / vth_perp**2 + dvz**2 / vth_perp**2

    # Kappa argument
    A_kappa = 1 + (v_par_sq + v_perp_sq) / (kappa - 1.5)

    # Normalization constant
    norm = (
        n * gamma(kappa + 1) /
        (gamma(kappa - 0.5) * (np.pi * (kappa - 1.5))**1.5 *
         vth_par * vth_perp**2)
    )

    f = norm * A_kappa**(-kappa - 1)
    
    vel_mag = np.linalg.norm(np.array([vx, vy, vz]), axis=0)
    E = 0.5 * sc.m_p * vel_mag * vel_mag
    
    counts = 2 * G * dt * (E**2) * f / (sc.m_p**2)
    
    return counts

def bi_kappa_bi_max(
    vx, vy, vz,
    n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k, kappa,
    n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m
):
    """
    Sum of bi-Kappa and bi-Maxwellian distributions.
    """
    f_kappa = bi_kappa(vx, vy, vz, n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k, kappa)
    f_maxwell = bi_Max(vx, vy, vz, n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m)
    
    return f_kappa + f_maxwell

def bi_max_bi_kappa(
    vx, vy, vz,
    n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k,
    n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m, kappa
):
    """
    Sum of bi-Kappa and bi-Maxwellian distributions.
    """
    f_kappa = bi_kappa(vx, vy, vz, n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m, kappa)
    f_maxwell = bi_Max(vx, vy, vz, n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k)
    
    return f_kappa + f_maxwell

def bi_kappa_bi_max_counts(
    vx, vy, vz,
    n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k, kappa,
    n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m,
    G, dt
):
    """
    Sum of bi-Kappa and bi-Maxwellian distributions.
    """
    f_kappa = bi_kappa(vx, vy, vz, n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k, kappa)
    f_maxwell = bi_Max(vx, vy, vz, n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m)
    
    f_comb = f_kappa + f_maxwell
    
    vel_mag = np.linalg.norm(np.array([vx, vy, vz]), axis=0)
    E = 0.5 * sc.m_p * vel_mag * vel_mag
    
    counts = 2 * G * dt * (E**2) * f_comb / (sc.m_p**2)
    
    return counts

def bi_max_bi_kappa_counts(
    vx, vy, vz,
    n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k,
    n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m, kappa,
    G, dt
):
    """
    Sum of bi-Kappa and bi-Maxwellian distributions.
    """
    f_kappa = bi_kappa(vx, vy, vz, n_m, vx_m, vy_m, vz_m, vth_par_m, vth_perp_m, kappa)
    f_maxwell = bi_Max(vx, vy, vz, n_k, vx_k, vy_k, vz_k, vth_par_k, vth_perp_k)
    
    f_comb = f_kappa + f_maxwell
    
    vel_mag = np.linalg.norm(np.array([vx, vy, vz]), axis=0)
    E = 0.5 * sc.m_p * vel_mag * vel_mag
    
    counts = 2 * G * dt * (E**2) * f_comb / (sc.m_p**2)
    
    return counts

def logp_double_kappa_minimisation(params, vx, vy, vz, G, dt, data):
        
    vals = params.valuesdict()
    
    # CORE parameters
    n1, vx1, vy1, vz1, vth_par1, vth_perp1, kappa1 = (
        vals['n_var1'], vals['vx1'], vals['vy1'], vals['vz1'],
        vals['v_th_par1'], vals['v_th_perp1'], vals['kappa1']
    )
    
    # BEAM parameters
    n2, vx2, vy2, vz2, vth_par2, vth_perp2 = (
        vals['n_var2'], vals['vx2'], vals['vy2'], vals['vz2'],
        vals['v_th_par2'], vals['v_th_perp2']
    )

    
    model = bi_kappa_bi_max_counts(
     vx, vy, vz,
     n1, vx1, vy1, vz1, vth_par1, vth_perp1, kappa1,
     n2, vx2, vy2, vz2, vth_par2, vth_perp2,
     G, dt
 )
    
    # start = time.time()
    
    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]
    
    # end = time.time()
    # time_test += end - start
    
    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)
    
    # end = time.time()
    # time_test += end - start

    # return negative of log likelihood
    return - log_likelihood

def logp_double_kappa_beam_minimisation(params, vx, vy, vz, G, dt, data):
        
    vals = params.valuesdict()
    
    # CORE parameters
    n1, vx1, vy1, vz1, vth_par1, vth_perp1 = (
        vals['n_var1'], vals['vx1'], vals['vy1'], vals['vz1'],
        vals['v_th_par1'], vals['v_th_perp1']
    )
    
    # BEAM parameters
    n2, vx2, vy2, vz2, vth_par2, vth_perp2, kappa2 = (
        vals['n_var2'], vals['vx2'], vals['vy2'], vals['vz2'],
        vals['v_th_par2'], vals['v_th_perp2'], vals['kappa2']
    )

    
    model = bi_max_bi_kappa_counts(
     vx, vy, vz,
     n1, vx1, vy1, vz1, vth_par1, vth_perp1,
     n2, vx2, vy2, vz2, vth_par2, vth_perp2, kappa2,
     G, dt
 )
    
    # start = time.time()
    
    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]
    
    # end = time.time()
    # time_test += end - start
    
    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)
    
    # end = time.time()
    # time_test += end - start

    # return negative of log likelihood
    return - log_likelihood

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

def goodness_of_fit(nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt):
    """    
    Calculate the goodness of fit for each set of parameters.
    """
    goodness_all = np.zeros(len(nc_all))

    for ind in tqdm(range(len(nc_all))):
                
        mean_ratio = goodness_of_fit_one(nc_all[ind], vc_all[ind], vth_par_c_all[ind], vth_perp_c_all[ind], nb_all[ind], vb_all[ind], vth_par_b_all[ind], vth_perp_b_all[ind], counts_in[ind], vx_bf[ind], vy_bf[ind], vz_bf[ind], G_factors[ind], dt)

        goodness_all[ind] = mean_ratio
    
    return goodness_all

def double_bi_Max_2D(upar, uperp, n_var1, vpar1, vperp1, v_th_par1, v_th_perp1, n_var2, vpar2, vperp2, v_th_par2, v_th_perp2):

    vel_par1 = upar - vpar1

    vel_perp1 = uperp - vperp1

    denominator1 = (np.pi ** 1.5) * v_th_par1 * v_th_perp1 * v_th_perp1
    term1 = n_var1 / denominator1
    exponent1 = ((vel_par1 * vel_par1) / (v_th_par1 * v_th_par1)) + \
        ((vel_perp1 * vel_perp1) / (v_th_perp1 * v_th_perp1))

    f1 = term1 * np.exp(- exponent1)

    vel_par2 = upar - vpar2

    vel_perp2 = uperp - vperp2

    denominator2 = (np.pi ** 1.5) * v_th_par2 * v_th_perp2 * v_th_perp2
    term2 = n_var2 / denominator2
    exponent2 = ((vel_par2 * vel_par2) / (v_th_par2 * v_th_par2)) + \
        ((vel_perp2 * vel_perp2) / (v_th_perp2 * v_th_perp2))

    f2 = term2 * np.exp(- exponent2)

    # if np.isfinite(f).any() == False:
    #    return 0

    return f1 + f2

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

def logp_kappa_minimisation_par_drift(params, vx, vy, vz, G, dt, data):
        
    vals = params.valuesdict()
    
    # CORE parameters
    n1, vx1, vy1, vz1, vth_par1, vth_perp1, kappa1 = (
        vals['n_var1'], vals['vx1'], vals['vy'], vals['vz'],
        vals['v_th_par1'], vals['v_th_perp1'], vals['kappa1']
    )
    
    # BEAM parameters
    n2, vx2, vy2, vz2, vth_par2, vth_perp2 = (
        vals['n_var2'], vals['vx2'], vals['vy'], vals['vz'],
        vals['v_th_par2'], vals['v_th_perp2']
    )

    
    model = bi_kappa_bi_max_counts(
     vx, vy, vz,
     n1, vx1, vy1, vz1, vth_par1, vth_perp1, kappa1,
     n2, vx2, vy2, vz2, vth_par2, vth_perp2,
     G, dt
 )
    
    # start = time.time()
    
    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]
    
    # end = time.time()
    # time_test += end - start
    
    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)
    
    # end = time.time()
    # time_test += end - start

    # return negative of log likelihood
    return - log_likelihood

def logp_kappa_beam_minimisation_par_drift(params, vx, vy, vz, G, dt, data):
        
    vals = params.valuesdict()
    
    # CORE parameters
    n1, vx1, vy1, vz1, vth_par1, vth_perp1 = (
        vals['n_var1'], vals['vx1'], vals['vy'], vals['vz'],
        vals['v_th_par1'], vals['v_th_perp1']
    )
    
    # BEAM parameters
    n2, vx2, vy2, vz2, vth_par2, vth_perp2, kappa2 = (
        vals['n_var2'], vals['vx2'], vals['vy'], vals['vz'],
        vals['v_th_par2'], vals['v_th_perp2'], vals['kappa2']
    )

    
    model = bi_max_bi_kappa_counts(
     vx, vy, vz,
     n1, vx1, vy1, vz1, vth_par1, vth_perp1,
     n2, vx2, vy2, vz2, vth_par2, vth_perp2, kappa2,
     G, dt
 )
    
    # start = time.time()
    
    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]
    
    # end = time.time()
    # time_test += end - start
    
    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)
    
    # end = time.time()
    # time_test += end - start

    # return negative of log likelihood
    return - log_likelihood

def logp_minimisation_kappa(params, vx, vy, vz, G, dt, data):

    vals = params.valuesdict()
    n_in = vals['n_var']
    vx_bulk_in = vals['vx']
    vy_bulk_in = vals['vy']
    vz_bulk_in = vals['vz']
    v_th_par_in = vals['v_th_par']
    v_th_perp_in = vals['v_th_perp']
    kappa_in = vals['kappa1']

    model = bi_kappa_counts(vx, vy, vz, n_in, vx_bulk_in, vy_bulk_in,
                                vz_bulk_in, v_th_par_in, v_th_perp_in, kappa_in, G, dt)

    # Apply a single combined mask
    mask = np.isfinite(data) & (model > 1e-50)
    data_masked = data[mask]
    model_masked = model[mask]

    log_likelihood = poisson_log_likelihood2(data_masked, model_masked)

    # return negative of log likelihood
    return - log_likelihood

def goodness_of_fit_one_kappa(nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt):
    """    
    Calculate the goodness of fit for each set of parameters.
    """
            
    n_fit_c = nc_all
    vx_fit_c = vc_all[0]
    vy_fit_c = vc_all[1]  
    vz_fit_c = vc_all[2]  
    v_th_par_fit_c = vth_par_c_all
    v_th_perp_fit_c = vth_perp_c_all
    kappa_c = kappa_all
    
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
    fit_params.add('kappa1', value=kappa_c)
    
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
    
    counts_fit = bi_kappa_bi_max_counts(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b,
                            vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b, G_in, dt)
    
    probs_data = logp_double_kappa_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_data_test)
    probs_model = logp_double_kappa_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_fit)

    # mean_ratio = np.sum(probs_data / probs_model) / (len(probs_data))
    mean_ratio =  probs_model / probs_data
    
    return mean_ratio

def goodness_of_fit_one_kappa_beam(nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, kappa_all, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt):
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
    kappa_b = kappa_all

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
    fit_params.add('kappa2', value=kappa_b)

    counts_data_test = counts_in.copy()
    dt = 1 / (9 * 96)
    G_in = np.copy(G_factors)
    
    mask_data = ~np.isfinite(G_in)
    G_in[mask_data] = np.nan
    counts_data_test[mask_data] = np.nan
    
    counts_fit = bi_max_bi_kappa_counts(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b,
                            vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b, kappa_b, G_in, dt)
    
    probs_data = logp_double_kappa_beam_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_data_test)
    probs_model = logp_double_kappa_beam_minimisation(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_fit)

    # mean_ratio = np.sum(probs_data / probs_model) / (len(probs_data))
    mean_ratio =  probs_model / probs_data
    
    return mean_ratio

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

def goodness_of_fit_one_core_only_kappa(nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all, counts_in, vx_bf, vy_bf, vz_bf, G_factors, dt):
    """    
    Calculate the goodness of fit for each set of parameters.
    """
            
    n_fit_c = nc_all
    vx_fit_c = vc_all[0]
    vy_fit_c = vc_all[1]  
    vz_fit_c = vc_all[2]  
    v_th_par_fit_c = vth_par_c_all
    v_th_perp_fit_c = vth_perp_c_all
    kappa_c = kappa_all

    fit_params = lmfit.Parameters()
    fit_params.add('n_var', value=n_fit_c)
    fit_params.add('vx', value=vx_fit_c)
    fit_params.add('vy', value=vy_fit_c)
    fit_params.add('vz', value=vz_fit_c)
    fit_params.add('v_th_par', value=v_th_par_fit_c)
    fit_params.add('v_th_perp', value=v_th_perp_fit_c)
    fit_params.add('kappa1', value=kappa_c)

    counts_data_test = counts_in.copy()
    dt = 1 / (9 * 96)
    G_in = np.copy(G_factors)
    
    mask_data = ~np.isfinite(G_in)
    G_in[mask_data] = np.nan
    counts_data_test[mask_data] = np.nan
    
    counts_fit = bi_kappa_counts(vx_bf, vy_bf, vz_bf, n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, G_in, dt)
    
    probs_data = logp_minimisation_kappa(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_data_test)
    probs_model = logp_minimisation_kappa(fit_params, vx_bf, vy_bf, vz_bf, G_in, dt, counts_fit)

    # mean_ratio = np.sum(probs_data / probs_model) / (len(probs_data))
    mean_ratio =  probs_model / probs_data
    
    return mean_ratio
    