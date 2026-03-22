# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:34:16 2025

@author: CI
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import matplotlib as mpl
import matplotlib.dates as mdates

from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import cmocean
from scipy.ndimage import convolve
import h5py

def Rotation_to_x(vector):
    """
    Produces a Rotation matrix that rotates the given vector such that it is 
    directed in the x-axis of a Cartesian grid.

    Parameters
    ----------
    vector : array
        The vector to be rotated.

    Returns
    -------
    R : ndarray
        The rotation matrix.

    """

    x_f = np.array((vector[0], vector[1], vector[2]) / np.sqrt((vector[0] * vector[0]) + \
                                        (vector[1] * vector[1]) + (vector[2] * vector[2])))
    x_r = np.array([1, 0, 0])
      
    y_f = np.cross(x_f, x_r)
    y_f = y_f / np.sqrt(np.dot(y_f, y_f))
    
    z_f = np.cross(x_f, y_f)
    z_f = z_f / np.sqrt(np.dot(z_f, z_f))
    
    R = np.array([x_f, y_f, z_f])
    
    return R

def E_to_v(E):
    """
    Turns energy (in eV) into velocity (in m/s) 

    Parameters
    ----------
    E : float/array
        The energy value(s).

    Returns
    -------
    v : float/array
        The velocity value(s).

    """
    v = np.sqrt(2 * E * sc.e / sc.m_p)
    return v

def v_to_E(v):
    """
    Turns velocity (in m/s) into Energy (in eV) 

    Parameters
    ----------
    E : float/array
        The energy value(s).

    Returns
    -------
    v : float/array
        The velocity value(s).

    """
    E = 0.5 * sc.m_p * v * v / sc.e
    return E

def SRF_rot(x, el, az):
    """
    Rotates a vector from the PAS instrument frame into the spacecraft frame.

    Parameters
    ----------
    x : array
        The vector to be rotated.
    el : float
        The elevation (theta) value.
    az : float
        The azimuth (phi) value.

    Returns
    -------
    y : array
        The rotated vector.

    """
    vx = x * - np.cos(el) * np.cos(az)
    vy = x * np.cos(el) * np.sin(az)
    vz = x * - np.sin(el)
    
    y = np.array([vx, vy, vz])
    return y

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

def Errors_f(vdf, counts):
    """
    Determines the errors of the VDF using the Counts level 1 data. The errors
    are calculated using df = (dC * f) / C (since df = dC / (G * dt * E * E))
    where dC is the Poisson error (sqrt(C)) of the counts.

    Parameters
    ----------
    vdf : array
        The Velocity Distribution Function as given from the level 2 data.
    counts : array
        The Counts measured by the PAS instrument as given from the level 1 data.

    Returns
    -------
    df : array
        The errors of the VDF.

    """
    
    # index = []

    # for i in range(len(t_vdf)):
    #     ind = np.where(np.logical_and(t_vdf[i] - 0.6 < t_counts, t_counts < t_vdf[i]))[0]
    #     index.append(ind[0])

    # index = np.array(index)
    
    df = np.zeros(vdf.shape)

    for i in range(len(vdf)):
    
        ind_vdf = np.where(vdf[i] > 0)
        count_val = counts[i][ind_vdf]
        vdf_val = vdf[i][ind_vdf]
    
    
        # df = (dC * f) / C (since df = dC / (G * dt * E * E))
        df[i][ind_vdf] = vdf_val / np.sqrt(count_val)
        
        ind_zero = np.where(vdf[i] == 0)
        # df_min = np.min(df[i][df[i] > 0])
        
        # df[i][ind_zero] = df_min
        df[i][ind_zero] = 0
    
        
    return df

def Bernoulli_integral(v, v_th, rho, B, gamma):
    """
    Calculates the Bernoulli integral of the plasma.

    Parameters
    ----------
    v : array
        The bulk velocity of the plasma.
    v_th : float
        The thermal velocity of the plasma.
    rho : float
        The mass density of the plasma.
    B : array
        The magnetic field of the plasma.
    gamma : float
        The polytropic index of the plasma.

    Returns
    -------
    B_int : float
        The Bernoulli integral of the plasma.

    """
    
    
    if 0.95 < gamma < 1.05:
        
        v_alfven_squared = 0.5 * B * B / (sc.mu_0 * rho)
        
        v_term = 0.5 * v * v
        
        v_thermal_term = 0.5 * v_th * v_th * np.log(rho)
        
        v_alfven_term = 0.5 * v_alfven_squared
        
        B_int = v_term + v_thermal_term + v_alfven_term
        
        return B_int
    
    else:
        v_alfven_squared = 0.5 * B * B / (sc.mu_0 * rho)
        
        v_term = 0.5 * v * v
        
        v_thermal_term = 0.5 * v_th * v_th * (gamma / (gamma - 1))
        
        v_alfven_term = 0.5 * v_alfven_squared
        
        B_int = v_term + v_thermal_term + v_alfven_term
        
        
        return B_int

def Get_G(vdf, counts, t_vdf, t_counts, phi, theta, energy):
    """
    Determines the G term of the instrument using the Counts level 1 data. G is
    calculated using G = C / (2 * E * E * dt * f), where dt is assumed to be 1 / (96*9).
    
    Parameters
    ----------
    vdf : array
        The Velocity Distribution Function as given from the level 2 data.
    counts : array
        The Counts measured by the PAS instrument as given from the level 1 data.
    t_vdf : array
        The time (in s) of the VDF data based on the VDF data time frame.
    t_counts : array
        The time (in s) of the count data based on the VDF data time frame.
    phi : array
        The azimuth bins of the instrument.
    theta : array
        The elevation bins of the instrument.
    energy : array
        The energy bins of the instrument.

    Returns
    -------
    G : array
        The value of the G term.

    """
    
    dt = 1 / (96 * 9)
    
    index = []

    for i in range(len(t_vdf)):
        ind = np.where(np.logical_and(t_vdf[i] - 0.6 < t_counts, t_counts < t_vdf[i]))[0]
        index.append(ind[0])

    index = np.array(index)
    
    G = np.zeros((len(t_vdf), len(phi), len(theta), len(energy)))

    for i in range(len(vdf)):
    
        ind_vdf = np.where(vdf[i] > 0)
        count_val = counts[index[i]][ind_vdf]
        vdf_val = vdf[i][ind_vdf]
        
        # G = C / (2 * E * E * dt * f)
        G[i][ind_vdf] = count_val / (2 * dt * vdf_val * ((energy[ind_vdf[2]]) ** 2))
    
    return G

def get_total_T_tensor(nc, vc, Tc, nb, vb, Tb):
    """
    Vectorized calculation of combined temperature tensor for time series inputs.

    Parameters:
    - Tc: (N,3,3) thermal temperature tensors of core
    - Tb: (N,3,3) thermal temperature tensors of beam
    - vc: (N,3) bulk velocity vectors of core
    - vb: (N,3) bulk velocity vectors of beam
    - nc: (N,) densities of core
    - nb: (N,) densities of beam

    Returns:
    - T_total: (N,3,3) combined temperature tensors for each time instance
    """

    n_total = nc + nb  # shape (N,)

    # Calculate total bulk velocity u_i for each time: shape (N,3)
    u = ((nc[:, None] * vc) + (nb[:, None] * vb)) / n_total[:, None]

    # Calculate drift velocities relative to u: shape (N,3)
    delta_v_core = vc - u
    delta_v_beam = vb - u

    # Drift kinetic tensors for core and beam: shape (N,3,3)
    drift_core = sc.m_p / sc.e * nc[:, None, None] * np.einsum('ni,nj->nij', delta_v_core, delta_v_core)
    drift_beam = sc.m_p / sc.e * nb[:, None, None] * np.einsum('ni,nj->nij', delta_v_beam, delta_v_beam)

    # Combine weighted thermal tensors and drift tensors: shape (N,3,3)
    T_total = (nc[:, None, None] * Tc + drift_core +
               nb[:, None, None] * Tb + drift_beam) / n_total[:, None, None]

    return T_total

def closest_datetime(target, datetime_array):
    """Find the closest datetime in the array to the target datetime."""
    datetime_array = np.array(datetime_array)  # Ensure it's a NumPy array
    time_diffs = np.abs(datetime_array - target)  # Compute absolute differences
    return np.argmin(time_diffs)  # Return closest datetime

def first_increasing_sequence_reverse(arr, threshold, length=4):
    """Finds the first instance (from the end) where `length` consecutive values are strictly increasing in a 1D array."""
    arr = np.array(arr)[::-1]  # Ensure input is a NumPy array and reverse it
    diffs = np.diff(arr) > 0  # Compute differences and check if increasing
    
    # Use a sliding window approach to check for `length-1` consecutive True values
    for i in range(threshold, len(diffs) - (length)):
        if np.all(diffs[i:i + (length - 1)]):
            return len(arr) - i - 1  # Return the starting index of the increasing sequence
    return None  # Return None if no such sequence is found

def first_four_consecutive_zeros_reverse(arr, threshold, length=4):
    """
    Finds the first instance (from the end) of `length` consecutive zeros in a 1D array after a given threshold index (from the end).

    Parameters:
    - arr (array-like): Input array.
    - threshold (int): The index after which to start searching.
    - length (int, optional): Number of consecutive zeros to search for (default is 4).

    Returns:
    - int: The index of the first occurrence after the threshold, or None if not found.
    """
    arr = np.array(arr)[::-1]  # Ensure input is a NumPy array and reverse it
    
    # Ensure threshold is within bounds
    if threshold >= len(arr) - length + 1:
        return None  # No valid sequence possible after threshold

    # Start searching from the index after the threshold
    for i in range(threshold, len(arr) - length + 1):
        if np.all(arr[i:i + length] == 0):
            return len(arr) - i - 1 # Return the first valid index

    return None  # Return None if no sequence is found

def calc_chi_square(vdf, errors, fit_vdf, f_pars):
    
    return sum(((vdf - fit_vdf) / errors)**2) / (len(vdf) - f_pars)

def integrate_vdf_over_angles(vdf, theta, errors=False):
    """Apply Jacobian and integrate over theta and phi."""
    
    if errors:
        weighted = (vdf * np.abs(np.cos(theta * np.pi / 180))[np.newaxis, :, np.newaxis])**2
        return np.sqrt(np.nansum(weighted, axis=(0, 1)))
    else:
        weighted = vdf * np.abs(np.cos(theta * np.pi / 180))[np.newaxis, :, np.newaxis]
        return np.nansum(weighted, axis=(0, 1))
    
def integrate_vdf_over_angles2(vdf, theta, phi, errors=False):
    """Apply Jacobian and integrate over theta and phi."""
    # Convert theta and phi to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # Calculate bin widths
    dtheta = np.mean(np.diff(theta_rad))
    dphi = np.mean(np.diff(phi_rad))

    if errors:
        weighted = (vdf * np.abs(np.cos(theta_rad))[np.newaxis, :, np.newaxis] * dtheta * dphi)**2 
        return np.sqrt(np.nansum(weighted, axis=(0, 1)))
    else:
        weighted = vdf * np.abs(np.cos(theta_rad))[np.newaxis, :, np.newaxis] * dtheta * dphi
        return np.nansum(weighted, axis=(0, 1))

def integrate_vdf_over_angles_trapz(vdf, theta, phi, errors = False):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    jacobian = np.cos(theta_rad)[np.newaxis, :, np.newaxis]
    if errors:
        weighted_vdf = (vdf * jacobian) ** 2
        vdf_phi = np.trapezoid(weighted_vdf, phi_rad, axis=0)
        vdf_theta_phi = np.trapezoid(vdf_phi, theta_rad, axis=0)
        return np.sqrt(vdf_theta_phi)
    else:
        weighted_vdf = vdf * jacobian
        vdf_phi = np.trapezoid(weighted_vdf, phi_rad, axis=0)
        vdf_theta_phi = np.trapezoid(vdf_phi, theta_rad, axis=0)
        return vdf_theta_phi  

def extract_all_fit_parameters(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                           nb_all, vb_all, vth_par_b_all, vth_perp_b_all, show = False):
    """Extract all necessary arrays for fit processing at index `ind`."""

    nc = nc_all[ind]
    vc = vc_all[ind]
    vth_par_c = vth_par_c_all[ind]
    vth_perp_c = vth_perp_c_all[ind]

    nb = nb_all[ind]
    vb = vb_all[ind]
    vth_par_b = vth_par_b_all[ind]
    vth_perp_b = vth_perp_b_all[ind]

    if show:
        print(f"nc: {nc*1e-6:.1f}, vc: [{vc[0]*1e-3:.1f}, {vc[1]*1e-3:.1f}, {vc[2]*1e-3:.1f}], "
      f"vth_par_c: {vth_par_c*1e-3:.1f}, vth_perp_c: {vth_perp_c*1e-3:.1f}")

        print(f"nb: {nb*1e-6:.1f}, vb: [{vb[0]*1e-3:.1f}, {vb[1]*1e-3:.1f}, {vb[2]*1e-3:.1f}], "
      f"vth_par_b: {vth_par_b*1e-3:.1f}, vth_perp_b: {vth_perp_b*1e-3:.1f}")

    return (nc, vc[0], vc[1], vc[2], vth_par_c, vth_perp_c,
            nb, vb[0], vb[1], vb[2], vth_par_b, vth_perp_b)

def extract_all_fit_parameters_core(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all):
    """Extract all necessary arrays for fit processing at index `ind`."""

    nc = nc_all[ind]
    vc = vc_all[ind]
    vth_par_c = vth_par_c_all[ind]
    vth_perp_c = vth_perp_c_all[ind]

    return (nc, vc[0], vc[1], vc[2], vth_par_c, vth_perp_c)

def extract_all_fit_parameters_kappa(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all,
                           nb_all, vb_all, vth_par_b_all, vth_perp_b_all, show=False):
    """Extract all necessary arrays for fit processing at index `ind`."""

    nc = nc_all[ind]
    vc = vc_all[ind]
    vth_par_c = vth_par_c_all[ind]
    vth_perp_c = vth_perp_c_all[ind]
    kappa_c = kappa_all[ind]

    nb = nb_all[ind]
    vb = vb_all[ind]
    vth_par_b = vth_par_b_all[ind]
    vth_perp_b = vth_perp_b_all[ind]

    if show:
        print(f"nc: {nc*1e-6:.1f}, vc: [{vc[0]*1e-3:.1f}, {vc[1]*1e-3:.1f}, {vc[2]*1e-3:.1f}], "
      f"vth_par_c: {vth_par_c*1e-3:.1f}, vth_perp_c: {vth_perp_c*1e-3:.1f}, kappa_c: {kappa_c:.1f}")

        print(f"nb: {nb*1e-6:.1f}, vb: [{vb[0]*1e-3:.1f}, {vb[1]*1e-3:.1f}, {vb[2]*1e-3:.1f}], "
      f"vth_par_b: {vth_par_b*1e-3:.1f}, vth_perp_b: {vth_perp_b*1e-3:.1f}")

    return (nc, vc[0], vc[1], vc[2], vth_par_c, vth_perp_c, kappa_c,
            nb, vb[0], vb[1], vb[2], vth_par_b, vth_perp_b)

def extract_all_fit_parameters_kappa_beam(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                           nb_all, vb_all, vth_par_b_all, vth_perp_b_all, kappa_all, show=False):
    """Extract all necessary arrays for fit processing at index `ind`."""

    nc = nc_all[ind]
    vc = vc_all[ind]
    vth_par_c = vth_par_c_all[ind]
    vth_perp_c = vth_perp_c_all[ind]

    nb = nb_all[ind]
    vb = vb_all[ind]
    vth_par_b = vth_par_b_all[ind]
    vth_perp_b = vth_perp_b_all[ind]
    kappa_b = kappa_all[ind]

    if show:
        print(f"nc: {nc*1e-6:.1f}, vc: [{vc[0]*1e-3:.1f}, {vc[1]*1e-3:.1f}, {vc[2]*1e-3:.1f}], "
      f"vth_par_c: {vth_par_c*1e-3:.1f}, vth_perp_c: {vth_perp_c*1e-3:.1f}")

        print(f"nb: {nb*1e-6:.1f}, vb: [{vb[0]*1e-3:.1f}, {vb[1]*1e-3:.1f}, {vb[2]*1e-3:.1f}], "
      f"vth_par_b: {vth_par_b*1e-3:.1f}, vth_perp_b: {vth_perp_b*1e-3:.1f}, kappa_b: {kappa_b:.1f}")

    return (nc, vc[0], vc[1], vc[2], vth_par_c, vth_perp_c,
            nb, vb[0], vb[1], vb[2], vth_par_b, vth_perp_b, kappa_b)

def define_pas_grid_parallel(theta, phi, energy, t_vdf, n, v_bulk, P_tensor, t_B, B, n_workers=32):
    ele, azi, ene = np.meshgrid(theta, phi, energy)
    speed = E_to_v(ene)

    vx = - speed * np.cos(ele * np.pi / 180) * np.cos(azi * np.pi / 180)
    vy = speed * np.cos(ele * np.pi / 180) * np.sin(azi * np.pi / 180)
    vz = - speed * np.sin(ele * np.pi / 180)

    temp_v = np.stack((vx, vy, vz), axis=0)

    # Outside the function — one-time conversion for performance
    t_B_np = np.array(t_B, dtype='datetime64[ms]')  # Efficient datetime format
    B_np = np.array(B)  # Ensure B is a NumPy array

    def process_rotation(i):
        t_i = t_vdf[i]
        v_bulk_i = v_bulk[i]
        P_tensor_i = P_tensor[i]

        # Convert t_i to NumPy datetime64 format
        t_i_np = np.datetime64(t_i, 'ms')
        t_min = t_i_np - np.timedelta64(500, 'ms')
        t_max = t_i_np + np.timedelta64(500, 'ms')

        # Use np.searchsorted to find the index range
        i_start = np.searchsorted(t_B_np, t_min, side='left')
        i_end = np.searchsorted(t_B_np, t_max, side='right')

        # Average magnetic field in the time window
        B_av = np.nanmean(B_np[i_start:i_end], axis=0)

        # Rotate and transform
        R_temp = Rotation_to_x(B_av)
        v_bulk_rot = np.matmul(R_temp, v_bulk_i * 1000)
        P_rot = R_temp @ P_tensor_i @ R_temp.T

        # Velocity components in new frame
        vx_i, vy_i, vz_i = np.tensordot(R_temp, temp_v, axes=(1, 0))

        return B_av, R_temp, v_bulk_rot, P_rot, vx_i, vy_i, vz_i


    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(process_rotation, range(len(t_vdf))), total=len(t_vdf)))

    B_all, R_all, v_bulk_bf_list, P_bf_list, vx_bf_list, vy_bf_list, vz_bf_list = zip(*results)

    B_all = np.array(B_all)
    v_bulk_bf = np.array(v_bulk_bf_list)
    P_bf = np.array(P_bf_list)
    vx_bf = np.array(vx_bf_list)
    vy_bf = np.array(vy_bf_list)
    vz_bf = np.array(vz_bf_list)

    T_par = (P_bf[:, 0, 0] / n[:]) / sc.e
    T_perp = ((P_bf[:, 1, 1] + P_bf[:, 2, 2]) / (2 * n[:])) / sc.e

    return B_all, R_all, v_bulk_bf, P_bf, vx_bf, vy_bf, vz_bf, T_par, T_perp

def define_pas_grid_parallel_T_only(theta, phi, energy, t_vdf, n, P_tensor, t_B, B, n_workers=32):
    ele, azi, ene = np.meshgrid(theta, phi, energy)
    speed = E_to_v(ene)

    vx = - speed * np.cos(ele * np.pi / 180) * np.cos(azi * np.pi / 180)
    vy = speed * np.cos(ele * np.pi / 180) * np.sin(azi * np.pi / 180)
    vz = - speed * np.sin(ele * np.pi / 180)

    temp_v = np.stack((vx, vy, vz), axis=0)

    # Outside the function — one-time conversion for performance
    t_B_np = np.array(t_B, dtype='datetime64[ms]')  # Efficient datetime format
    B_np = np.array(B)  # Ensure B is a NumPy array

    def process_rotation(i):
        t_i = t_vdf[i]
        # v_bulk_i = v_bulk[i]
        P_tensor_i = P_tensor[i]

        # Convert t_i to NumPy datetime64 format
        t_i_np = np.datetime64(t_i, 'ms')
        t_min = t_i_np - np.timedelta64(500, 'ms')
        t_max = t_i_np + np.timedelta64(500, 'ms')

        # Use np.searchsorted to find the index range
        i_start = np.searchsorted(t_B_np, t_min, side='left')
        i_end = np.searchsorted(t_B_np, t_max, side='right')

        # Average magnetic field in the time window
        B_av = np.nanmean(B_np[i_start:i_end], axis=0)

        # Rotate and transform
        R_temp = Rotation_to_x(B_av)
        # v_bulk_rot = np.matmul(R_temp, v_bulk_i * 1000)
        P_rot = R_temp @ P_tensor_i @ R_temp.T

        # Velocity components in new frame
        # vx_i, vy_i, vz_i = np.tensordot(R_temp, temp_v, axes=(1, 0))

        # return B_av, R_temp, v_bulk_rot, P_rot, vx_i, vy_i, vz_i
        return B_av, R_temp, P_rot

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(process_rotation, range(len(t_vdf))), total=len(t_vdf)))

    # B_all, R_all, v_bulk_bf_list, P_bf_list, vx_bf_list, vy_bf_list, vz_bf_list = zip(*results)
    B_all, R_all, P_bf_list = zip(*results)

    B_all = np.array(B_all)
    # v_bulk_bf = np.array(v_bulk_bf_list)
    P_bf = np.array(P_bf_list)
    # vx_bf = np.array(vx_bf_list)
    # vy_bf = np.array(vy_bf_list)
    # vz_bf = np.array(vz_bf_list)

    T_par = (P_bf[:, 0, 0] / n[:]) / sc.e
    T_perp = ((P_bf[:, 1, 1] + P_bf[:, 2, 2]) / (2 * n[:])) / sc.e

    # return B_all, R_all, v_bulk_bf, P_bf, vx_bf, vy_bf, vz_bf, T_par, T_perp
    return B_all, R_all, P_bf, T_par, T_perp

def average_B_moments(t_vdf, t_B, B, n_workers=32):

    # Outside the function — one-time conversion for performance
    t_B_np = np.array(t_B, dtype='datetime64[ms]')  # Efficient datetime format
    B_np = np.array(B)  # Ensure B is a NumPy array

    def process_average(i):
        t_i = t_vdf[i]
        # Convert t_i to NumPy datetime64 format
        t_i_np = np.datetime64(t_i, 'ms')
        t_min = t_i_np - np.timedelta64(500, 'ms')
        t_max = t_i_np + np.timedelta64(500, 'ms')

        # Use np.searchsorted to find the index range
        i_start = np.searchsorted(t_B_np, t_min, side='left')
        i_end = np.searchsorted(t_B_np, t_max, side='right')

        # Average magnetic field in the time window
        B_av = np.nanmean(B_np[i_start:i_end], axis=0)

        return B_av

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(process_average, range(len(t_vdf))), total=len(t_vdf)))

    B_all = results

    B_all = np.array(B_all)

    return B_all

def Check_count_shape(t_vdf, t_l1, vdf, counts):
    t_vdf_np = np.array(t_vdf, dtype='datetime64[ms]')
    t_l1_np = np.array(t_l1, dtype='datetime64[ms]')

    if t_vdf_np.shape == t_l1_np.shape:
        print('Count and VDF shapes match!!!')
        return counts

    print('Count and VDF shapes do not match, filling counts based on VDF time stamps...')

    counts_old = np.copy(counts)
    counts_new = np.zeros_like(vdf)

    for i, t_i in enumerate(t_vdf_np):
        t_offset = t_i - np.timedelta64(505, 'ms')  # 0.6 seconds
        # Find the first index where t_l1 > t_offset using searchsorted
        start_idx = np.searchsorted(t_l1_np, t_offset, side='right')
        end_idx = np.searchsorted(t_l1_np, t_i, side='left')

        if start_idx < end_idx:
            if end_idx - start_idx > 1:
                print(f"Warning: Multiple points found in window for VDF index {i} (count: {end_idx - start_idx})")
            counts_new[i] = counts_old[start_idx]
    return counts_new

def average_1count(counts, vdf, dt):
    counts_nans = np.copy(counts).astype(float)
    counts_nans[counts == 65535] = np.nan  # counts array with filled values (65535) turned into nans

    c1_mean = np.nanmean(vdf / counts_nans, axis=0)  # mean 1 count vdf value per pixel
    c1_max = np.nanmax(vdf / counts_nans, axis=0)  # max 1 count vdf value per pixel

    # Test for one pixel
    # pixel_test_1c = np.zeros(len(vdf))
    # for i in range(len(vdf)):

    #     pixel_test_1c[i] = vdf[i, 5, 5, 40] / counts_nans[i, 5, 5, 40]

    # hist_1c = plt.hist(pixel_test_1c, bins = 20, color = blue)
    # plt.vlines(c1_mean[5, 5, 40], ymin=0, ymax = max(hist_1c[0]), color = orange, label = 'mean')
    # plt.vlines(c1_max[5, 5, 40], ymin=0, ymax = max(hist_1c[0]), color = green, label = 'max')
    # plt.xlabel('VDF / C')
    # plt.show()

    return(c1_mean, c1_max, counts_nans)

def counts_fill(vdf, counts, c1_max):
    vdf_counts_check = vdf
    c1_max_t = np.broadcast_to(c1_max, vdf.shape)

    mask_fill_counts = (counts == 65535)
    mask_fill_vdf = mask_fill_counts & (vdf_counts_check != 0)

    if np.any(mask_fill_counts):
        if np.any(mask_fill_vdf):
            print("""
            *********************************************************************************************************
            Datapoints found where VDF != 0 and Counts has filled value. Filled values replaced using 1 count average
            *********************************************************************************************************
            """)
            counts[mask_fill_vdf] = np.round(vdf[mask_fill_vdf] / c1_max_t[mask_fill_vdf])
            # After processing, set any remaining filled counts to 0
            # remaining_fills = (counts == 65535)
            # if np.any(remaining_fills):
            #     counts[remaining_fills] = 0.0
            counts[mask_fill_counts & (counts == 65535)] = 0.0

        else:
            print("""
            *****************************************************************************************
            Datapoints found where VDF = 0 and Counts has filled value. Filled values replaced with 0
            *****************************************************************************************
            """)
            counts[mask_fill_counts] = 0.0
    else:
        print("""
        ********************************************************************************
        ***************** No datapoints found Counts has filled value ******************
        ********************************************************************************
        """) 

    return counts

def fill_zero_gaps(vdf, counts_nans, energy, dt):

    start_time = datetime.now()
    
    G_factors = counts_nans * (sc.m_p**2) / (2 * dt * vdf * ((energy * sc.e)**2))  # G factor
    G_factors_og = np.copy(G_factors)
    G_factors[~np.isfinite(G_factors)] = np.nan  # Replace non-finite values with NaN
    G_mean = np.nanmean(G_factors, axis=0)  # Average G factor for each pixel
    rem_infs = ~np.isfinite(G_mean)
    G_mean[rem_infs] = np.nan

    "Plotting G factors for a specific pixel"
    # fig, ax = plt.subplots(ncols=2, figsize = [50, 20])
    # ax[0].hist(G_factors[:, 5, 5, 50], bins = 20)
    # ax[0].set_xlabel('G')
    # ax[0].set_ylabel('Occurrence')

    # ax[1].plot(t_vdf, G_factors[:, 5, 5, 50])
    # ax[1].set_xlabel('G')
    # plt.suptitle(f'G at $\Theta$ = {theta[5]:.2f}, $\Phi$ = {phi[5]:.2f}, E = {energy[50]:.0f} eV', fontsize = 50)
    # plt.show()

    # file = "fit_results\\2022_02_28\\G_mean"
    # np.save(file, G_mean)
    # G_mean_day = np.load('fit_results\\2022_02_28\\G_mean.npy', allow_pickle=True)
    # G_mean = G_mean_day

    "Replace non-finite values (0 measurement) with mean G value"
    G_factors_updated = np.where(np.isfinite(G_factors), G_factors, G_mean) 

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    return G_factors_updated, G_factors_og, G_mean

def fill_zero_gaps_sliding_mean(vdf, counts_nans, energy, dt, times, window=timedelta(days=1), block_cols=None):
    """
    Fill invalid (NaN/Inf) G_factors at each time row with the mean of that
    pixel over a ±window around that time, computed fully vectorized.

    Parameters
    ----------
    vdf : np.ndarray, shape (N, ..., E)
        VDF values (time is axis 0; energy is assumed to be the last axis).
    counts_nans : np.ndarray, same shape as vdf
        Counts array with possible NaNs.
    energy : np.ndarray, shape (E,)
        Energy bins corresponding to the last axis of vdf.
    dt : float
        Time width per measurement (seconds).
    times : array-like of datetime.datetime, shape (N,)
        Sorted times aligned with vdf along axis=0.
    window : datetime.timedelta
        Half-window size (default ±1 day).
    block_cols : int or None
        If set, process trailing pixels in blocks of this many columns to reduce memory.

    Returns
    -------
    G_factors_filled : np.ndarray, same shape as vdf
        G_factors with invalid entries filled by local sliding means.
    G_factors_original : np.ndarray, same shape as vdf
        Original G_factors (NaNs retained).
    G_mean_global : np.ndarray, shape of trailing dims (..., E)
        Global mean across time (mainly for diagnostics).
    """
    start_time = datetime.now()

    # Determine G factors
    den = (energy * sc.e) ** 2
    den = den.reshape((1,) * (vdf.ndim - 1) + (-1,))  # (..., E) on last axis
    G = counts_nans * (sc.m_p ** 2) / (2 * dt * vdf * den)
    G_original = np.copy(G)

    # Mark invalid as NaN
    G = np.where(np.isfinite(G), G, np.nan)

    # Prepare time instances inside window where mean will take place
    times_np = np.asarray(times, dtype="datetime64[ns]")
    w = np.timedelta64(int(window.total_seconds()), "s")
    t_start = times_np - w
    t_end   = times_np + w

    # determines start and end timestamps of the window for each timestamp
    start_idx = np.searchsorted(times_np, t_start, side="left")
    end_idx   = np.searchsorted(times_np, t_end,   side="right")

    # Flatten array to time axis and a combined axis of theta, phi and energy (M).
    N = G.shape[0]
    trailing_shape = G.shape[1:]
    M = int(np.prod(trailing_shape)) # number of total point per time instance
    G2 = G.reshape(N, M)  # (N, M)

    # Fills invalid G values with the mean G value 
    def process_block(G_block):
        # G_block: (N, m)
        valid = np.isfinite(G_block)
        X = np.where(valid, G_block, 0.0).astype(G_block.dtype, copy=False)
        V = valid.astype(np.int64)

        Sx = np.cumsum(X, axis=0) # sum of all values 
        Sv = np.cumsum(V, axis=0)

        # Pad with a zero row so range [i,j): sum = S[i] - S[j]
        Sx_pad = np.vstack([np.zeros((1, G_block.shape[1]), dtype=X.dtype), Sx])
        Sv_pad = np.vstack([np.zeros((1, G_block.shape[1]), dtype=V.dtype), Sv])

        # Broadcast start/end indices to (N, m) shape
        m = G_block.shape[1]
        start2 = np.broadcast_to(start_idx[:, None], (N, m))
        end2   = np.broadcast_to(end_idx[:,   None], (N, m))

        # Determine sum of G values and number of points in the block
        sum_end   = np.take_along_axis(Sx_pad, end2,   axis=0)
        sum_start = np.take_along_axis(Sx_pad, start2, axis=0)
        cnt_end   = np.take_along_axis(Sv_pad, end2,   axis=0)
        cnt_start = np.take_along_axis(Sv_pad, start2, axis=0)

        window_sum = sum_end - sum_start
        window_cnt = cnt_end - cnt_start

        # Avoid divide-by-zero: keep NaN where count==0
        with np.errstate(invalid="ignore", divide="ignore"):
            local_mean = window_sum / np.where(window_cnt > 0, window_cnt, 1)

        # Fill only bad entries at each row/pixel
        filled = np.where(valid, G_block, local_mean)

        return filled

    # Perform calculation in one go, or in blocks of columns to save memory
    if block_cols is None:
        G2_filled = process_block(G2)
    else:
        G2_filled = np.empty_like(G2)
        for c0 in range(0, M, block_cols):
            c1 = min(M, c0 + block_cols)
            G2_filled[:, c0:c1] = process_block(G2[:, c0:c1])

    G_filled = G2_filled.reshape((N,) + trailing_shape)

    # Global mean (diagnostic)
    G_mean_global = np.nanmean(G_filled, axis=0)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    return G_filled, G_original, G_mean_global

def kmeans_filter_parallel(vdf, counts, theta, energy, n_workers=16):
    """
    Parallel KMeans filtering to remove alpha particles from VDF data.
    
    Parameters:
    - vdf: (N, 11, 9, 96) array
    - counts: same shape as vdf
    - theta: 1D array of elevation angles
    - energy: 1D array of energy bins
    
    Returns:
    - vdf_in: VDF with NaN filtering
    - vdf_in_0: VDF with zero filtering
    - counts_in: counts with NaN filtering
    - counts_in_0: counts with zero filtering
    - ind_f_all: first index array
    - ind_e_all: end index array
    """
    vdf_in = np.copy(vdf)
    vdf_in_0 = np.copy(vdf)
    counts_in = np.copy(counts).astype(float)
    counts_in_0 = np.copy(counts).astype(float)
    ind_f_all = np.zeros(len(vdf), dtype=int)
    ind_e_all = np.zeros(len(vdf), dtype=int)
    vels = E_to_v(energy) * 1e-3

    def process_vdf(i):

        vdf_i = vdf[i]
        counts_i = counts[i]
        vdf_temp = vdf_i * abs(np.cos(theta * np.pi / 180))[np.newaxis, :, np.newaxis]
        vdf_temp = np.sum(vdf_temp, axis=(0, 1))
        vdf_temp = np.where(vdf_temp == 0, np.nan, vdf_temp)
        vdf_log = np.log10(vdf_temp)

        vdf_log_filt = np.copy(vdf_log)

        try:
            first_ind = first_increasing_sequence_reverse(vdf_log, 0, length=4)
            if first_ind is None:
                ind_max = np.nanargmax(vdf_log_filt)
                first_ind = np.where(energy < energy[ind_max]*0.6)[0][0]
                ind_f_all[i] = first_ind
                vdf_log_filt[first_ind:] = np.nan
            
            else:
                ind_f_all[i] = first_ind
                vdf_log_filt[first_ind+1:] = np.nan

        except Exception as e:
            print(f"First Error at index {i}: {e}")
            return None
        
        try:
            try:
                der_vdf = np.gradient(vdf_log_filt, vels)
                mask = np.isfinite(vels) & np.isfinite(der_vdf) & np.isfinite(vdf_log_filt)
                data = np.column_stack((vels[mask], der_vdf[mask], vdf_log[mask]))

                kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
                labels = kmeans.labels_

                first_value = labels[0]
                first_diff = np.where(labels != first_value)[0][0]
                vel_break = data[first_diff, 0]
                end_ind = np.argmin(np.abs(vels - vel_break))
                ind_e_all[i] = end_ind

                return i, first_ind, end_ind
            
            except Exception as e:
                # print(f"ValueError at index {i}: {e}")
                ind_max = np.nanargmax(vdf_log_filt)
                end_ind = np.where(energy > energy[ind_max]*1.5)[0][-1] + 1
                ind_e_all[i] = end_ind

                return i, first_ind, end_ind

        except Exception as e:
            print(f"End Error at index {i}: {e}")

            return None

    # Parallel execution
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(process_vdf, range(len(vdf))), total=len(vdf)))

    for res in results:
        if res is None:
            continue
        i, f, e = res
        vdf_in_0[i, :, :, f:] = 0
        vdf_in_0[i, :, :, :e] = 0
        counts_in_0[i, :, :, f:] = 0
        counts_in_0[i, :, :, :e] = 0

        vdf_in[i, :, :, f:] = np.nan
        vdf_in[i, :, :, :e] = np.nan
        counts_in[i, :, :, f:] = np.nan
        counts_in[i, :, :, :e] = np.nan

    return vdf_in, vdf_in_0, counts_in, counts_in_0, ind_f_all, ind_e_all

def plot_E_time_series(vdf, vdf_in, t_vdf, energy, theta, phi):
    "PLOT THE ENERGY TIME SERIES"
    # First integrate the VDF along theta and phi
    # Apply Jacobian of cos(theta) to VDF
    vdf_E_plot = vdf * abs(np.cos(theta * np.pi / 180))[np.newaxis, np.newaxis, :, np.newaxis]
    # Sum (integrate) over theta and phi
    values = np.nansum(vdf_E_plot, axis=(1, 2))
    values = np.log10(values)  # take log10.

    # Apply Jacobian of cos(theta) to filtered VDF
    vdf_E_plot_filt = vdf_in * abs(np.cos(theta * np.pi / 180))[np.newaxis, np.newaxis, :, np.newaxis]
    # Sum (integrate) over theta and phi
    values_filt = np.nansum(vdf_E_plot_filt, axis=(1, 2))
    values_filt = np.log10(values_filt)  # take log10.

    # Create color plot of VDF function across energy and time.
    fig, ax = plt.subplots(nrows=2)
    current_cmap = cmocean.cm.thermal

    c1 = ax[0].pcolormesh(t_vdf, np.log10(energy), values.T, cmap=current_cmap)
    c2 = ax[1].pcolormesh(t_vdf, np.log10(energy), values_filt.T, cmap=current_cmap)
    cbar1 = fig.colorbar(c1, ax=ax[0], norm=mpl.colors.LogNorm())
    cbar1.set_label('$log_{10}$(f, Integrated over Φ and Θ)')
    cbar2 = fig.colorbar(c2, ax=ax[1], norm=mpl.colors.LogNorm())
    cbar2.set_label('$log_{10}$(f, Integrated over Φ and Θ)')
    # plt.xlabel()
    ax[0].set_ylabel('$log_{10}$(Energy / Charge [eV])')
    ax[1].set_ylabel('$log_{10}$(Energy / Charge [eV])')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax[0].set_xticks([])
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

    ax[0].set_title('Unfiltered')
    ax[1].set_title('Filtered')

    plt.show()

def combine_fit_params(nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                       nb_all, vb_all, vth_par_b_all, vth_perp_b_all, B_mag = np.nan, n = np.nan):
    """
    Combines fit parameters for core and beam populations into a single structure.
    """

    # Compute temperatures for core
    T_par_c = sc.m_p * (vth_par_c_all**2) / 2 / sc.e  # in eV
    T_perp_c = sc.m_p * (vth_perp_c_all**2) / 2 / sc.e  # in eV
    T_mag_c = (T_par_c + T_perp_c + T_perp_c) / 3  # in eV#

    v_mag_c = np.linalg.norm(vc_all, axis=1)  # in m/s
    vc_par_vec = np.copy(vc_all)
    vc_par_vec[:, 1:] = 0
    vc_perp_vec = np.copy(vc_all)
    vc_perp_vec[:, 0] = 0

    # Create T tensor for core
    Tc_tensor = np.zeros([len(T_par_c), 3, 3])
    Tc_tensor[:, 0, 0] = T_par_c  # in eV
    Tc_tensor[:, 1, 1], Tc_tensor[:, 2, 2] = T_perp_c, T_perp_c  # in eV

    # Compute temperatures for beam
    T_par_b = sc.m_p * (vth_par_b_all**2) / 2 / sc.e  # in eV
    T_perp_b = sc.m_p * (vth_perp_b_all**2) / 2 / sc.e  # in eV
    T_mag_b = (T_par_b + T_perp_b + T_perp_b) / 3  # in eV

    v_mag_b = np.linalg.norm(vb_all, axis=1)  # in m/s
    vb_par_vec = np.copy(vb_all)
    vb_par_vec[:, 1:] = 0
    vb_perp_vec = np.copy(vb_all)
    vb_perp_vec[:, 0] = 0

    # Create T tensor for beam
    Tb_tensor = np.zeros([len(T_par_b), 3, 3])
    Tb_tensor[:, 0, 0] = T_par_b  # in eV
    Tb_tensor[:, 1, 1], Tb_tensor[:, 2, 2] = T_perp_b, T_perp_b  # in eV

    # Compute total properties
    total_n = (nc_all + nb_all) * 1e-6  # in cm^-3
    total_v = 1e-3 * ((nc_all[:, np.newaxis] * vc_all) + (nb_all[:, np.newaxis] * vb_all)) / (nc_all[:, np.newaxis] + nb_all[:, np.newaxis])  # in km/s

    total_T_tensor = get_total_T_tensor(nc_all, vc_all, Tc_tensor, nb_all, vb_all, Tb_tensor)  # in eV
    total_T = (total_T_tensor[:, 0, 0] + total_T_tensor[:, 1, 1] + total_T_tensor[:, 2, 2]) / 3
    total_T_par = total_T_tensor[:, 0, 0]
    total_T_perp = (total_T_tensor[:, 1, 1] + total_T_tensor[:, 2, 2]) / 2

    "Drift speed"
    v_drift = np.linalg.norm(vb_all - vc_all, axis = 1) * 1e-3 # in km/s
    v_drift_par = abs(vb_all[:, 0] - vc_all[:, 0]) * 1e-3 # in km/s
    # v_drift_perp = np.linalg.norm(v_vec_perp_c - v_vec_perp_b) * 1e-3 # in km/s
    VA = (B_mag * 1e-9 / np.sqrt(sc.mu_0 * sc.m_p * n*1e6)) * 1e-3 # in km/s

    "T ratio between core and beam"
    T_ratio = T_mag_b / T_mag_c # T ratio between beam and core
    n_ratio = nb_all / nc_all # n ratio between beam and core

    "Beta"
    beta_par = 2 * sc.mu_0 * total_n * 1e6 * total_T_par * sc.e / (B_mag * B_mag * 1e-18)
    beta_perp = 2 * sc.mu_0 * total_n * 1e6 * total_T_perp * sc.e / (B_mag * B_mag * 1e-18)
    "T anisotropy (T_perp / T_par)"
    a_p = total_T_perp / total_T_par

    return (T_par_c, T_perp_c, T_mag_c, T_par_b, T_perp_b, T_mag_b, total_n, total_v, total_T, total_T_par, total_T_perp, v_drift_par, VA, n_ratio, T_ratio, beta_par, beta_perp, a_p)

def combine_fit_params2(nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                       nb_all, vb_all, vth_par_b_all, vth_perp_b_all, B_mag = np.nan, n = np.nan):
    """
    Combines fit parameters for core and beam populations into a single structure.
    """

    # Compute temperatures for core
    T_par_c = sc.m_p * (vth_par_c_all**2) / 2 / sc.e  # in eV
    T_perp_c = sc.m_p * (vth_perp_c_all**2) / 2 / sc.e  # in eV
    T_mag_c = (T_par_c + T_perp_c + T_perp_c) / 3  # in eV#

    v_mag_c = np.linalg.norm(vc_all, axis=1)  # in m/s
    vc_par_vec = np.copy(vc_all)
    vc_par_vec[:, 1:] = 0
    vc_perp_vec = np.copy(vc_all)
    vc_perp_vec[:, 0] = 0

    # Create T tensor for core
    Tc_tensor = np.zeros([len(T_par_c), 3, 3])
    Tc_tensor[:, 0, 0] = T_par_c  # in eV
    Tc_tensor[:, 1, 1], Tc_tensor[:, 2, 2] = T_perp_c, T_perp_c  # in eV

    # Compute temperatures for beam
    T_par_b = sc.m_p * (vth_par_b_all**2) / 2 / sc.e  # in eV
    T_perp_b = sc.m_p * (vth_perp_b_all**2) / 2 / sc.e  # in eV
    T_mag_b = (T_par_b + T_perp_b + T_perp_b) / 3  # in eV

    v_mag_b = np.linalg.norm(vb_all, axis=1)  # in m/s
    vb_par_vec = np.copy(vb_all)
    vb_par_vec[:, 1:] = 0
    vb_perp_vec = np.copy(vb_all)
    vb_perp_vec[:, 0] = 0

    # Create T tensor for beam
    Tb_tensor = np.zeros([len(T_par_b), 3, 3])
    Tb_tensor[:, 0, 0] = T_par_b  # in eV
    Tb_tensor[:, 1, 1], Tb_tensor[:, 2, 2] = T_perp_b, T_perp_b  # in eV

    # Compute total properties
    total_n = (nc_all + nb_all)
    total_v = ((nc_all[:, np.newaxis] * vc_all) + (nb_all[:, np.newaxis] * vb_all)) / (nc_all[:, np.newaxis] + nb_all[:, np.newaxis])

    total_T_tensor = get_total_T_tensor(nc_all, vc_all, Tc_tensor, nb_all, vb_all, Tb_tensor)  # in eV
    total_T = (total_T_tensor[:, 0, 0] + total_T_tensor[:, 1, 1] + total_T_tensor[:, 2, 2]) / 3
    total_T_par = total_T_tensor[:, 0, 0]
    total_T_perp = (total_T_tensor[:, 1, 1] + total_T_tensor[:, 2, 2]) / 2

    "Drift speed"
    v_drift = np.linalg.norm(vb_all - vc_all, axis = 1)
    v_drift_par = abs(vb_all[:, 0]) - abs(vc_all[:, 0])
    # v_drift_perp = np.linalg.norm(v_vec_perp_c - v_vec_perp_b) * 1e-3 # in km/s
    VA = (B_mag * 1e-9 / np.sqrt(sc.mu_0 * sc.m_p * n)) * 1e-3 # in km/s

    "T ratio between core and beam"
    T_ratio = T_mag_b / T_mag_c # T ratio between beam and core
    n_ratio = nb_all / nc_all # n ratio between beam and core

    "Beta"
    beta_par = 2 * sc.mu_0 * total_n * total_T_par * sc.e / (B_mag * B_mag * 1e-18)
    beta_perp = 2 * sc.mu_0 * total_n * total_T_perp * sc.e / (B_mag * B_mag * 1e-18)
    "T anisotropy (T_perp / T_par)"
    a_p = total_T_perp / total_T_par

    return (T_par_c, T_perp_c, T_mag_c, T_par_b, T_perp_b, T_mag_b, total_n, total_v, total_T, total_T_par, total_T_perp, v_drift_par, VA, n_ratio, T_ratio, beta_par, beta_perp, a_p)

def remove_isolated_points(i, counts_in, vdf_in, n_points = 0):
    counts_in_test = np.copy(counts_in[i])
    vdf_in_test = np.copy(vdf_in[i])

    "Remove isolated points (only checking 6 direct neighboors)"
    # Count valid points before
    before = np.sum((~np.isnan(counts_in_test)) & (counts_in_test != 0))

    # Assume `data` is your (11, 9, 96) array
    # Define a 3D kernel for 6-connected neighbors (no diagonals)
    kernel = np.zeros((3, 3, 3), dtype=int)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1

    # Create mask of valid (non-zero, non-NaN) values
    valid_mask = (~np.isnan(counts_in_test)) & (counts_in_test != 0)

    # Count valid neighbors using convolution
    neighbor_count = convolve(valid_mask.astype(int), kernel, mode='constant', cval=0)

    # Mask for isolated pixels (valid but with 0 valid neighbors)
    isolated_mask = (valid_mask & (neighbor_count <= n_points))

    # Replace isolated pixels with 0
    counts_in_test[isolated_mask] = np.nan
    vdf_in_test[isolated_mask] = np.nan

    # Count valid points after
    after = np.sum((~np.isnan(counts_in_test)) & (counts_in_test != 0))

    # Print how many points were removed
    # print(f"Number of isolated points removed: {before - after}")

    return i, counts_in_test, vdf_in_test

def remove_isolated_points_parallel(counts_in, vdf_in, n_points=0, n_workers=8):
    counts_out = np.copy(counts_in)
    vdf_out = np.copy(vdf_in)

    def worker(i):
        return remove_isolated_points(i, counts_in, vdf_in, n_points)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(worker, range(len(counts_in))), total=len(counts_in)))

    for i, c_out, v_out in results:
        counts_out[i] = c_out
        vdf_out[i] = v_out

    return counts_out, vdf_out

def combine_core_beam(nc_p, vc_p, vth_par_c_p, vth_perp_c_p, nb_p, vb_p, vth_par_b_p, vth_perp_b_p):

    N = len(nc_p)  # Number of time steps

    # Compute temperatures for core
    T_par_c_p = sc.m_p * (vth_par_c_p**2) / 2 / sc.e  # in eV
    T_perp_c_p = sc.m_p * (vth_perp_c_p**2) / 2 / sc.e  # in eV
    T_mag_c_p = (T_par_c_p + T_perp_c_p + T_perp_c_p) / 3  # in eV#

    v_mag_c_p = np.linalg.norm(vc_p, axis=1)  # in m/s
    vc_par_vec_p = np.copy(vc_p)
    vc_par_vec_p[:, 1:] = 0
    vc_par_p = vc_par_vec_p[:, 0]
    vc_perp_vec_p = np.copy(vc_p)
    vc_perp_vec_p[:, 0] = 0
    vc_perp_p = np.linalg.norm(vc_perp_vec_p, axis = 1)

    # Create T tensor for core
    Tc_tensor_p = np.zeros([N, 3, 3])
    Tc_tensor_p[:, 0, 0] = T_par_c_p  # in eV
    Tc_tensor_p[:, 1, 1], Tc_tensor_p[:, 2, 2] = T_perp_c_p, T_perp_c_p  # in eV

    # Compute temperatures for beam
    T_par_b_p = sc.m_p * (vth_par_b_p**2) / 2 / sc.e  # in eV
    T_perp_b_p = sc.m_p * (vth_perp_b_p**2) / 2 / sc.e  # in eV
    T_mag_b_p = (T_par_b_p + T_perp_b_p + T_perp_b_p) / 3  # in eV

    v_mag_b_p = np.linalg.norm(vb_p, axis=1)  # in m/s
    vb_par_vec_p = np.copy(vb_p)
    vb_par_vec_p[:, 1:] = 0
    vb_par_p = vb_par_vec_p[:, 0]
    vb_perp_vec_p = np.copy(vb_p)
    vb_perp_vec_p[:, 0] = 0
    vb_perp_p = np.linalg.norm(vb_perp_vec_p, axis = 1)

    # Create T tensor for beam
    Tb_tensor_p = np.zeros([N, 3, 3])
    Tb_tensor_p[:, 0, 0] = T_par_b_p  # in eV
    Tb_tensor_p[:, 1, 1], Tb_tensor_p[:, 2, 2] = T_perp_b_p, T_perp_b_p  # in eV

    # Compute total properties
    total_n_p = (nc_p + nb_p) * 1e-6  # in cm^-3
    total_v_p = 1e-3 * ((nc_p[:, np.newaxis] * vc_p) + (nb_p[:, np.newaxis] * vb_p)) / (nc_p[:, np.newaxis] + nb_p[:, np.newaxis])  # in km/s
    v_p = np.linalg.norm(total_v_p, axis = 1)

    total_T_tensor_p = get_total_T_tensor(nc_p, vc_p, Tc_tensor_p, nb_p, vb_p, Tb_tensor_p)  # in eV
    total_T_p = (total_T_tensor_p[:, 0, 0] + total_T_tensor_p[:, 1, 1] + total_T_tensor_p[:, 2, 2]) / 3
    total_T_par_p = total_T_tensor_p[:, 0, 0]
    total_T_perp_p = (total_T_tensor_p[:, 1, 1] + total_T_tensor_p[:, 2, 2]) / 2

    return (total_n_p, total_v_p, vc_par_p, vc_perp_p, vb_par_p, vb_perp_p, total_v_p, v_p, T_par_c_p, T_perp_c_p, T_par_b_p, T_perp_b_p, total_T_par_p, total_T_perp_p, total_T_p)

def has_duplicates(arr):
    """
    Returns True if there are any duplicates in arr, False otherwise.
    Fast for large arrays.
    """
    arr = np.asarray(arr)
    return np.unique(arr).size != arr.size

def find_duplicate_indices(arr):
    """
    Returns a list of indices of all duplicate entries in arr.
    Fast for large arrays.
    """
    arr = np.asarray(arr)
    # Use a dictionary for fast lookup
    seen = {}
    dup_indices = []
    for idx, val in enumerate(arr):
        if val in seen:
            dup_indices.append(idx)
        else:
            seen[val] = idx
    return dup_indices

def duplicate_mask(arr):
    """
    Returns a boolean array where True indicates a duplicate entry in arr.
    Fast for large arrays.
    Only the second and later occurrences are marked True.
    """
    arr = np.asarray(arr)
    # Use numpy's unique with return_index for speed
    _, idx_first = np.unique(arr, return_index=True)
    mask = np.ones(arr.shape, dtype=bool)
    mask[idx_first] = False
    return mask

def apply_filter_mask(filter_mask, *arrays):
    """
    Applies a boolean filter mask to each input array.
    Accepts any number of arrays and returns them filtered in the same order.
    
    Parameters:
    - filter_mask: boolean array
    - *arrays: any number of arrays to be filtered
    
    Returns:
    - tuple of filtered arrays
    """
    return tuple(arr[filter_mask] for arr in arrays)

def vth_to_T(vth_par, vth_perp):
    """
    Convert thermal speeds to temperatures.
    
    Parameters:
    - vth_par: parallel thermal speed (m/s)
    - vth_perp: perpendicular thermal speed (m/s)
    
    Returns:
    - T_par: parallel temperature (eV)
    - T_perp: perpendicular temperature (eV)
    - T_mag: average temperature (eV)
    """
    T_par = sc.m_p * (vth_par**2) / (2 * sc.e)  # in eV
    T_perp = sc.m_p * (vth_perp**2) / (2 * sc.e)  # in eV
    T_mag = (T_par + 2 * T_perp) / 3  # in eV

    return T_par, T_perp, T_mag

def find_matching_indices(arr1, arr2):
    """
    Find indices in arr1 where values match any value in arr2.
    Both arrays must be sorted and contain datetime objects.
    Returns a NumPy array of indices in arr1 where arr1[i] == arr2[j] for some j.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    idx = np.searchsorted(arr1, arr2)
    # Remove out-of-bounds and non-matches
    mask = (idx < len(arr1)) & (arr1[idx] == arr2)
    return idx[mask]

def moving_average_time_window(times, data, window_seconds=60):
    """
    Fast moving average over a time window for large, sorted arrays.
    times: array of datetime.datetime or np.datetime64 (must be sorted)
    data: array of same length as times
    window_seconds: window size in seconds (default 60)
    Returns: array of moving averages (same length as data)
    """
    times_np = np.array(times)
    if np.issubdtype(times_np.dtype, np.datetime64):
        times_np = times_np.astype('datetime64[s]').astype('int64')
    else:
        # Assume datetime.datetime
        times_np = np.array([int(t.timestamp()) for t in times_np])

    data = np.asarray(data)
    N = len(times_np)
    result = np.full(N, np.nan, dtype=float)
    half_window = window_seconds // 2

    # Use searchsorted for fast window indexing
    for i in range(N):
        t_min = times_np[i] - half_window
        t_max = times_np[i] + half_window
        left = np.searchsorted(times_np, t_min, side='left')
        right = np.searchsorted(times_np, t_max, side='right')
        if right > left:
            result[i] = np.nanmean(data[left:right])
    return result

def moving_average_time_window_multi(time_array, *data_arrays, window_seconds=60):
    """
    Moving average over a time window for multiple arrays (all same length), robust for irregular time grids.
    Returns a tuple of arrays, each averaged over the moving window.
    If a window is empty, the result is NaN for that index.
    Output arrays are always the same size as the input arrays.
    """
    times = np.asarray(time_array)
    N = len(times)
    # Convert times to seconds since epoch for fast searchsorted
    if np.issubdtype(times.dtype, np.datetime64):
        times_sec = times.astype('datetime64[s]').astype('int64')
    else:
        times_sec = np.array([int(t.timestamp()) for t in times])

    half_window = window_seconds // 2
    t_min = times_sec - half_window
    t_max = times_sec + half_window

    left_idx = np.searchsorted(times_sec, t_min, side='left')
    right_idx = np.searchsorted(times_sec, t_max, side='right')

    results = []
    for arr in data_arrays:
        arr = np.asarray(arr)
        out = np.full(N, np.nan)
        for i in range(N):
            window = arr[left_idx[i]:right_idx[i]]
            if window.size > 0:
                out[i] = np.nanmean(window)
        results.append(out)
    return tuple(results)

def moving_average_time_window_multi_parallel(time_array, *data_arrays, window_seconds=60, n_workers=8):
    """
    Parallel moving average over a time window for multiple arrays (all same length).
    Returns a tuple of arrays, each averaged over the moving window.
    If a window is empty, the result is NaN for that index.
    Output arrays are always the same size as the input arrays.
    """
    times = np.asarray(time_array)
    N = len(times)
    if np.issubdtype(times.dtype, np.datetime64):
        times_sec = times.astype('datetime64[s]').astype('int64')
    else:
        times_sec = np.array([int(t.timestamp()) for t in times])

    half_window = window_seconds // 2
    data_arrays = [np.asarray(arr) for arr in data_arrays]

    def compute_avg(i):
        t_min = times_sec[i] - half_window
        t_max = times_sec[i] + half_window
        left = np.searchsorted(times_sec, t_min, side='left')
        right = np.searchsorted(times_sec, t_max, side='right')
        avgs = []
        for arr in data_arrays:
            if right > left:
                try:
                    avgs.append(np.nanmean(arr[left:right]))
                except Exception:
                    avgs.append(np.nan)
            else:
                avgs.append(np.nan)
        return avgs

    results = np.full((N, len(data_arrays)), np.nan, dtype=float)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i, avgs in enumerate(executor.map(compute_avg, range(N))):
            results[i, :] = avgs

    return tuple(results[:, j] for j in range(results.shape[1]))
    
def moving_average_time_window_multi_vectorised(time_array, *data_arrays, window_seconds=60):
    """
    Fully vectorized moving average over a time window for multiple arrays (all same length).
    Returns a tuple of arrays, each averaged over the moving window.
    If a window is empty, the result is NaN for that index.
    Output arrays are always the same size as the input arrays.
    """
    import numpy as np

    times = np.asarray(time_array)
    N = len(times)
    # Convert times to seconds since epoch for fast searchsorted
    if np.issubdtype(times.dtype, np.datetime64):
        times_sec = times.astype('datetime64[s]').astype('int64')
    else:
        times_sec = np.array([int(t.timestamp()) for t in times])

    half_window = window_seconds // 2

    # Precompute window bounds for all indices
    t_min = times_sec - half_window
    t_max = times_sec + half_window

    # For each i, find left and right window indices
    left_idx = np.searchsorted(times_sec, t_min, side='left')
    right_idx = np.searchsorted(times_sec, t_max, side='right')

    results = []
    for arr in data_arrays:
        arr = np.asarray(arr)
        # Compute cumulative sum and count of finite values
        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, 0.0)
        cumsum = np.concatenate([[0], np.cumsum(arr_filled)])
        count = np.concatenate([[0], np.cumsum(valid.astype(int))])

        # Window sum and count for each index
        window_sum = cumsum[right_idx] - cumsum[left_idx]
        window_count = count[right_idx] - count[left_idx]

        # Avoid divide by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            avg = window_sum / window_count
        avg[window_count == 0] = np.nan
        results.append(avg)

    return tuple(results)

def save_moments_to_h5(
    filename,
    t_mom,
    n,
    T,
    v_bulk,
    P_tensor,
    T_par,
    T_perp,
    R_all
):

    t_mom_str = np.array([d.isoformat() for d in t_mom], dtype="S32")

    with h5py.File(filename, "a") as f:
        def save_or_append(name, data):
            shape = data.shape
            maxshape = (None,) + shape[1:]
            if name in f:
                dset = f[name]
                old_size = dset.shape[0]
                new_size = old_size + shape[0]
                dset.resize((new_size,) + shape[1:])
                dset[old_size:new_size] = data
            else:
                f.create_dataset(
                    name, data=data, maxshape=maxshape, chunks=True, compression="gzip"
                )

        save_or_append("t_mom", t_mom_str)
        save_or_append("n", np.array(n))
        save_or_append("T", np.array(T))
        save_or_append("v_bulk", np.array(v_bulk))
        save_or_append("P_tensor", np.array(P_tensor))
        save_or_append("T_par", np.array(T_par))
        save_or_append("T_perp", np.array(T_perp))
        save_or_append("R_all", np.array(R_all))

def load_moments_from_h5(filename):
    """
    Load moments arrays from an HDF5 file.
    t_mom will be returned as an array of Python datetime objects.
    Returns:
        t_mom, n, T, v_bulk, P_tensor, T_par, T_perp, R_all
    """

    with h5py.File(filename, "r") as f:
        t_mom_str = np.array(f["t_mom"]).astype(str)
        # Convert ISO8601 strings back to datetime objects
        t_mom = np.array([datetime.fromisoformat(s) for s in t_mom_str])
        n = np.array(f["n"])
        T = np.array(f["T"])
        v_bulk = np.array(f["v_bulk"])
        P_tensor = np.array(f["P_tensor"])
        T_par = np.array(f["T_par"])
        T_perp = np.array(f["T_perp"])
        R_all = np.array(f["R_all"])
    return t_mom, n, T, v_bulk, P_tensor, T_par, T_perp, R_all

def save_B_to_h5(
    filename,
    t_mom,
    B
):

    t_mom_str = np.array([d.isoformat() for d in t_mom], dtype="S32")

    with h5py.File(filename, "a") as f:
        def save_or_append(name, data):
            shape = data.shape
            maxshape = (None,) + shape[1:]
            if name in f:
                dset = f[name]
                old_size = dset.shape[0]
                new_size = old_size + shape[0]
                dset.resize((new_size,) + shape[1:])
                dset[old_size:new_size] = data
            else:
                f.create_dataset(
                    name, data=data, maxshape=maxshape, chunks=True, compression="gzip"
                )

        save_or_append("t_B", t_mom_str)
        save_or_append("B", np.array(B))

def load_B_from_h5(filename):
    """
    Load B array from an HDF5 file.
    t_B will be returned as an array of Python datetime objects.
    Returns:
        t_B, B
    """

    with h5py.File(filename, "r") as f:
        t_B_str = np.array(f["t_B"]).astype(str)
        # Convert ISO8601 strings back to datetime objects
        t_B = np.array([datetime.fromisoformat(s) for s in t_B_str])
        B = np.array(f["B"])

    return t_B, B

def save_ks_to_h5(
    filename,
    t_ks,
    ks,
    p
):

    t_ks_str = np.array([d.isoformat() for d in t_ks], dtype="S32")

    with h5py.File(filename, "a") as f:
        def save_or_append(name, data):
            shape = data.shape
            maxshape = (None,) + shape[1:]
            if name in f:
                dset = f[name]
                old_size = dset.shape[0]
                new_size = old_size + shape[0]
                dset.resize((new_size,) + shape[1:])
                dset[old_size:new_size] = data
            else:
                f.create_dataset(
                    name, data=data, maxshape=maxshape, chunks=True, compression="gzip"
                )

        save_or_append("t_ks", t_ks_str)
        save_or_append("ks", np.array(ks))
        save_or_append("p", np.array(p))

def load_ks_from_h5(filename):
    """
    Load B array from an HDF5 file.
    t_B will be returned as an array of Python datetime objects.
    Returns:
        t_B, B
    """

    with h5py.File(filename, "r") as f:
        t_ks_str = np.array(f["t_ks"]).astype(str)
        # Convert ISO8601 strings back to datetime objects
        t_ks = np.array([datetime.fromisoformat(s) for s in t_ks_str])
        ks = np.array(f["ks"])
        p = np.array(f["p"])

    return t_ks, ks, p