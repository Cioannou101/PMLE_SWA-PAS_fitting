"""
Author: Charalambos Ioannou
Institution: UCL / MSSL
Email: charalambos.ioannou.22@ucl.ac.uk
GitHub: @Cioannou101
Created: 2026-06-07

This script contains general useful functions used across the project, especially in the pre-processing of the VDF data.
It includes vector rotations, VDF calculations, error calculations, and defining the PAS grid in the magnetic field frame.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import matplotlib as mpl
import matplotlib.dates as mdates

from tqdm import tqdm
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

def first_increasing_sequence_reverse(arr, threshold, length=4):
    """Finds the first instance (from the end) where `length` consecutive values are strictly increasing in a 1D array."""
    arr = np.array(arr)[::-1]  # Ensure input is a NumPy array and reverse it
    diffs = np.diff(arr) > 0  # Compute differences and check if increasing
    
    # Use a sliding window approach to check for `length-1` consecutive True values
    for i in range(threshold, len(diffs) - (length)):
        if np.all(diffs[i:i + (length - 1)]):
            return len(arr) - i - 1  # Return the starting index of the increasing sequence
    return None  # Return None if no such sequence is found

def integrate_vdf_over_angles(vdf, theta, errors=False):
    """Apply Jacobian and integrate over theta and phi."""
    
    if errors:
        weighted = (vdf * np.abs(np.cos(theta * np.pi / 180))[np.newaxis, :, np.newaxis])**2
        return np.sqrt(np.nansum(weighted, axis=(0, 1)))
    else:
        weighted = vdf * np.abs(np.cos(theta * np.pi / 180))[np.newaxis, :, np.newaxis]
        return np.nansum(weighted, axis=(0, 1))
     
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
    """Extract all necessary arrays for fit processing at index `ind`, core only."""

    nc = nc_all[ind]
    vc = vc_all[ind]
    vth_par_c = vth_par_c_all[ind]
    vth_perp_c = vth_perp_c_all[ind]

    return (nc, vc[0], vc[1], vc[2], vth_par_c, vth_perp_c)

def define_pas_grid_parallel(theta, phi, energy, t_vdf, n, v_bulk, P_tensor, t_B, B, n_workers=32):
    """
    Defines the PAS grid in local magnetic field frame. [1, 0, 0] is parallel to the magnetic field. 
    """
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

def fill_zero_gaps_sliding_mean(vdf, counts_nans, energy, dt, times, window=timedelta(days=1), block_cols=None):
    """
    Fill invalid (NaN/Inf) G_factors at each time row with the mean of that
    pixel over a ±window around that time.

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
    "PLOT THE UNFILTERED AND FILTERED ENERGY TIME SERIES"
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

