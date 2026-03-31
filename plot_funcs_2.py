# -*- coding: utf-8 -*-
"""
Created on Mon May  5 12:09:40 2025

@author: CI
"""
import os
# vdf_processing.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from gen_funcs_2 import *
from Poisson_fit_functions_2 import double_bi_Max, bi_Max, bi_kappa, bi_kappa_bi_max, bi_max_bi_kappa
from scipy.stats import gaussian_kde
# from datetime import datetime

def save_figure(fig, folder_path, filename):
    save_path = os.path.join(folder_path, filename)
    fig.savefig(save_path)
    plt.close(fig)

def plot_integrated_vdf(ax, vels, core_f, beam_f, data_f, fit_f, errors_f, vc_mag, t_vdf_ind, lims = [1e-10, 1e-6], core_only = False):

    if core_only == True:
    
        ax[1].plot(vels, np.log10(core_f), '--', color='teal', label='core', lw=2)
        ax[1].errorbar(vels, np.log10(data_f), yerr=errors_f/(data_f * np.log(10)), fmt='x', ms=5, mew=2, color='steelblue', label='data')
        # ax[1].errorbar(vels, np.log10(data_f), fmt='x', ms=12, mew=4, color='steelblue', label='data')
        ax[1].set_ylim(bottom=-10, top=-6.5)
        ax[1].set_xlabel("V (km/s)")
        ax[1].set_ylabel("VDF")
        ax[1].legend(loc=1, frameon=False)
        ax[1].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[1].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')
    
        ax[0].plot(vels, core_f, '--', color='teal', label='core', lw=2)
        ax[0].errorbar(vels, data_f, yerr=errors_f, fmt='x', ms=5, mew=2, color='steelblue', label='data')
        
        ax[0].set_xlabel("V (km/s)")
        ax[0].set_ylabel("VDF")
        ax[0].legend(loc=1, frameon=False)
        ax[0].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[0].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')
        
    else:
        ax[1].plot(vels, core_f, '--', color='teal', label='core', lw=2)
        ax[1].plot(vels, beam_f, '--', color='#CC5500', label='beam', lw=2)
        ax[1].errorbar(vels, data_f, yerr=errors_f, fmt='x', ms=4, mew=2, color='steelblue', label='data')
        # ax[1].errorbar(vels, np.log10(data_f), fmt='x', ms=12, mew=4, color='steelblue', label='data')
        ax[1].plot(vels, fit_f, color='red', label='fit', lw=2)
        ax[1].set_ylim(bottom=lims[0], top=lims[1])
        ax[1].set_xlabel("V (km/s)")
        ax[1].set_ylabel("VDF")
        ax[1].legend(loc=1, frameon=False)
        ax[1].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[1].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')
        ax[1].set_yscale('log')

        ax[0].plot(vels, core_f, '--', color='teal', label='core', lw=2)
        ax[0].plot(vels, beam_f, '--', color='#CC5500', label='beam', lw=2)
        ax[0].errorbar(vels, data_f, yerr=errors_f, fmt='x', ms=4, mew=2, color='steelblue', label='data')
        ax[0].plot(vels, fit_f, color='red', label='fit', lw=2)

        ax[0].set_xlabel("V (km/s)")
        ax[0].set_ylabel("VDF")
        ax[0].legend(loc=1, frameon=False)
        ax[0].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[0].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')

def plot_energy_grid(energy, vdf_in, vdf_fit, core_fit, beam_fit,
                     errors_in, ind, theta, phi, p_ran=5, lims = [1e-10, 1e-6], core_only = False):
    """
    Plot linear and log-scale energy-space VDF components near peak over angular region.
    Produces two separate figures.
    """

    # Identify peak angular region
    az_ind, el_ind, _ = np.unravel_index(np.nanargmax(vdf_in), vdf_in.shape)
    az_range = np.clip(np.arange(az_ind - p_ran // 2, az_ind + p_ran // 2 + 1), 0, vdf_in.shape[0] - 1)
    el_range = np.clip(np.arange(el_ind - p_ran // 2, el_ind + p_ran // 2 + 1), 0, vdf_in.shape[1] - 1)

    # Create figures
    fig_lin, ax_lin = plt.subplots(p_ran, p_ran, figsize=(60, 40), sharex=True, sharey=True)
    fig_log, ax_log = plt.subplots(p_ran, p_ran, figsize=(60, 40), sharex=True, sharey=True)
    
    if core_only == True:
        for k, az in enumerate(az_range):
            for j, el in enumerate(el_range):
                data = vdf_in[az, el]
                mask = np.isfinite(data)
        
                if not np.any(mask):
                    continue  # skip if all values are NaN
        
                e = energy[mask]
                data_vals = data[mask]
                core_vals = core_fit[az, el, mask]
                err_vals = errors_in[az, el, mask]
        
                # Linear plot
                ax_lin[k, j].errorbar(e, data_vals, yerr=err_vals, fmt='o', ms=12, mew=3, color='steelblue', label='data')
                ax_lin[k, j].plot(e, core_vals, '--', color='teal', lw=3, label='core')
                ax_lin[k, j].set_ylim(top=np.nanmax(vdf_in) * 1.1, bottom=0)
                ax_lin[k, j].legend()
        
                # Log plot
                ax_log[k, j].errorbar(e, np.log10(data_vals), fmt='o', ms=12, mew=3, color='steelblue', label='data')
                ax_log[k, j].plot(e, np.log10(core_vals), '--', color='teal', lw=3, label='core')
                ax_log[k, j].set_ylim(top=-7, bottom=-10.5)
                ax_log[k, j].legend()

        # Set axis labels and titles
        for i in range(p_ran):
            ax_lin[p_ran - 1, i].set_xlabel('V (km/s)')
            ax_log[p_ran - 1, i].set_xlabel('V (km/s)')
            ax_lin[0, i].set_title(f'$\\Theta = {theta[el_range[i]]:.1f}^\\circ$')
            ax_log[0, i].set_title(f'$\\Theta = {theta[el_range[i]]:.1f}^\\circ$')
            ax_lin[i, 0].set_ylabel(f'VDF - $\\phi = {phi[az_range[i]]:.1f}^\\circ$')
            ax_log[i, 0].set_ylabel(f'VDF - $\\phi = {phi[az_range[i]]:.1f}^\\circ$')
        
    else:
        for k, az in enumerate(az_range):
            for j, el in enumerate(el_range):
                data = vdf_in[az, el]
                mask = np.isfinite(data)
        
                if not np.any(mask):
                    continue  # skip if all values are NaN
        
                e = E_to_v(energy[mask])*1e-3
                data_vals = data[mask]
                fit_vals = vdf_fit[az, el, mask]
                core_vals = core_fit[az, el, mask]
                beam_vals = beam_fit[az, el, mask]
                err_vals = errors_in[az, el, mask]
        
                # Linear plot
                ax_lin[k, j].errorbar(e, data_vals, yerr=err_vals, fmt='o', ms=12, mew=3, color='steelblue', label='data')
                ax_lin[k, j].plot(e, core_vals, '--', color='teal', lw=3, label='core')
                ax_lin[k, j].plot(e, beam_vals, '--', color='#CC5500', lw=3, label='beam')
                ax_lin[k, j].plot(e, fit_vals, '-', color='darkred', lw=3, label='fit')
                ax_lin[k, j].set_ylim(top=np.nanmax(vdf_in) * 1.1, bottom=0)
                ax_lin[k, j].legend()
        
                # Log plot
                ax_log[k, j].errorbar(e, data_vals, yerr=err_vals, fmt='o', ms=12, mew=3, color='steelblue', label='data')
                ax_log[k, j].plot(e, core_vals, '--', color='teal', lw=3, label='core')
                ax_log[k, j].plot(e, beam_vals, '--', color='#CC5500', lw=3, label='beam')
                ax_log[k, j].plot(e, fit_vals, '-', color='darkred', lw=3, label='fit')
                ax_log[k, j].set_ylim(top=lims[1], bottom=lims[0])
                ax_log[k, j].set_yscale('log')
                ax_log[k, j].legend()
    
        # Set axis labels and titles
        for i in range(p_ran):
            ax_lin[p_ran - 1, i].set_xlabel('V (km/s)')
            ax_log[p_ran - 1, i].set_xlabel('V (km/s)')
            ax_lin[0, i].set_title(f'$\\Theta = {theta[el_range[i]]:.1f}^\\circ$')
            ax_log[0, i].set_title(f'$\\Theta = {theta[el_range[i]]:.1f}^\\circ$')
            ax_lin[i, 0].set_ylabel(f'VDF - $\\phi = {phi[az_range[i]]:.1f}^\\circ$')
            ax_log[i, 0].set_ylabel(f'VDF - $\\phi = {phi[az_range[i]]:.1f}^\\circ$')

    return fig_lin, fig_log

def plot_fits(plot_indices, parameters, vdf_in, errors_in, vx_bf, vy_bf, vz_bf, date1_str, t_vdf, theta, phi, energy, folder, lims=[-10., -6.5], plot_3D = False, save = False, p_ran=5, core_only = False):
    
    if core_only:
        nc_all, vc_all, vth_par_c_all, vth_perp_c_all = parameters
    else:
        nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all = parameters
    "PLOT FITS"
    p_ran = p_ran  # Plotting range parameter

    vels = E_to_v(energy)*1e-3  # Convert energy to velocity

    errors = errors_in

    # Create plot directory if not exist
    if save==True:
        folder_path = os.path.join('Plots', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', folder, 'Example_fits')
        os.makedirs(folder_path, exist_ok=True)

    for ind in tqdm(plot_indices):
        data_f = vdf_in[ind]
        errors_f = errors[ind]
        # theta_slice = theta_all[ind]
        if core_only:
            (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c) = extract_all_fit_parameters_core(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all)
        
        else:
            (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c,
                n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b) = extract_all_fit_parameters(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                                                                                                                    nb_all, vb_all, vth_par_b_all, vth_perp_b_all, show = False)

        vc_mag = np.linalg.norm([vx_fit_c, vy_fit_c, vz_fit_c]) * 1e-3

        # Model VDFs
        if core_only:
            vdf_fit = np.nan
            core_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
            beam_fit = np.nan
        
        else:
            vdf_fit = double_bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b,
                                    vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)
            core_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
            beam_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind],
                            n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)
        # az_range, el_range = get_index_ranges(vdf, p_ran)

        # Prepare velocity data for integration
        if core_only:
            data_integrated = integrate_vdf_over_angles(data_f, theta)
            fit_integrated = np.nan
            core_integrated = integrate_vdf_over_angles(core_fit, theta)
            beam_integrated = np.nan
            errors_integrated = integrate_vdf_over_angles(errors_f, theta, errors= True)

        else:
            data_integrated = integrate_vdf_over_angles(data_f, theta)
            fit_integrated = integrate_vdf_over_angles(vdf_fit, theta)
            core_integrated = integrate_vdf_over_angles(core_fit, theta)
            beam_integrated = integrate_vdf_over_angles(beam_fit, theta)
            errors_integrated = integrate_vdf_over_angles(errors_f, theta, errors= True)
        
        # Plotting
        fig, ax = plt.subplots(ncols=2)

        plot_integrated_vdf(
            ax, vels, core_integrated, beam_integrated, data_integrated, 
            fit_integrated, errors_integrated, lims = lims, vc_mag=vc_mag, t_vdf_ind=t_vdf[ind], core_only=core_only
            )
        
        # Save figure
        if save == True:
            filename = f'1d_{ind}.png'
            save_figure(fig, folder_path, filename)
        else:
            plt.show()
        
        if plot_3D == True:

            fig_lin, fig_log = plot_energy_grid(
            energy, data_f, vdf_fit, core_fit, beam_fit,
            errors_f, ind, theta, phi, lims = lims, p_ran=p_ran, core_only=core_only
            )
            
            # Save figure
            if save == True:
                filename_log = f'grid_log_{ind}.png'
                filename_lin = f'grid_{ind}.png'
                save_figure(fig_log, folder_path, filename_log)
                save_figure(fig_lin, folder_path, filename_lin)

            else:
                plt.show(fig_lin)
                plt.show(fig_log)

def plot_fits2(plot_indices, parameters, vdf_in, errors_in, vx_bf, vy_bf, vz_bf, date1_str, t_vdf, theta, phi, energy, folder, lims=[-10., -6.5], plot_3D = False, save = False, p_ran=5):
    
    nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all = parameters
    "PLOT FITS"
    p_ran = p_ran  # Plotting range parameter

    vels = E_to_v(energy)*1e-3  # Convert energy to velocity

    errors = errors_in

    # Create plot directory if not exist

    folder_path = os.path.join('Plots', 'Poisson_fits_data', f'{date1_str[0]}_{date1_str[1]}_{date1_str[2]}', 'double_bi_max', folder, 'Example_fits')
    os.makedirs(folder_path, exist_ok=True)

    for ind in tqdm(plot_indices):
        data_f = vdf_in[ind]
        errors_f = errors[ind]
        # theta_slice = theta_all[ind]
        
        (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c,
        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b) = extract_all_fit_parameters(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                                                                                                            nb_all, vb_all, vth_par_b_all, vth_perp_b_all, show = False)
        vc_mag = np.linalg.norm([vx_fit_c, vy_fit_c, vz_fit_c]) * 1e-3

        # Model VDFs
        # vdf_fit = double_bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b,
        #                         vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)
        vdf_fit = np.zeros_like(data_f)
        core_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
        beam_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind],
                        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)
        
        # az_range, el_range = get_index_ranges(vdf, p_ran)

        # Prepare velocity data for integration
        data_integrated = integrate_vdf_over_angles(data_f, theta)
        fit_integrated = integrate_vdf_over_angles(vdf_fit, theta)
        core_integrated = integrate_vdf_over_angles(core_fit, theta)
        beam_integrated = integrate_vdf_over_angles(beam_fit, theta)
        errors_integrated = integrate_vdf_over_angles(errors_f, theta, errors= True)
        
        # Plotting
        fig, ax = plt.subplots(ncols=2, figsize=(40, 15))

        plot_integrated_vdf(
            ax, vels, core_integrated, beam_integrated, data_integrated, 
            fit_integrated, errors_integrated, lims = lims, vc_mag=vc_mag, t_vdf_ind=t_vdf[ind]
            )
        
        # Save figure
        if save == True:
            filename = f'1d_{ind}.png'
            save_figure(fig, folder_path, filename)
        else:
            plt.show()
        
        if plot_3D == True:

            fig_lin, fig_log = plot_energy_grid(
            energy, data_f, vdf_fit, core_fit, beam_fit,
            errors_f, ind, theta, phi, lims = lims, p_ran=p_ran
            )
            
            # Save figure
            if save == True:
                filename_log = f'grid_log_{ind}.png'
                filename_lin = f'grid_{ind}.png'
                save_figure(fig_log, folder_path, filename_log)
                save_figure(fig_lin, folder_path, filename_lin)

            else:
                plt.show(fig_lin)
                plt.show(fig_log)
        
    return core_fit, beam_fit

def plot_combined_time_series(t_vdf, n, total_n, total_v, total_T, v_bulk_bf, vc_all, vb_all, nc_all, nb_all, T, T_mag_c, T_mag_b):
    """
    Plot combined time series of moments and fit parameters.
    """

    fig, ax = plt.subplots(nrows = 3, figsize = [30, 20])

    # ax[0].plot(t_bia[::100], n_bia[::100], color = 'darkgreen', label = 'rpw')
    ax[0].plot(t_vdf, nc_all*1e-6, color = 'teal', label = 'core')
    ax[0].plot(t_vdf, nb_all*1e-6, color = '#CC5500', label = 'beam')
    ax[0].plot(t_vdf, total_n, color = 'darkred', label = 'fit total')
    ax[0].plot(t_vdf, n, '.', color = 'black', label = 'moments')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax[0].legend(loc = 4)
    ax[0].set_ylabel('$n (cm^{-3})$')
    ax[0].set_xticks([])

    ax[1].plot(t_vdf, np.linalg.norm(vc_all*1e-3, axis=1), color = 'teal', label = 'core')
    ax[1].plot(t_vdf, np.linalg.norm(vb_all*1e-3, axis=1), color = '#CC5500', label = 'beam')
    ax[1].plot(t_vdf, np.linalg.norm(total_v, axis=1), color = 'darkred', label = 'fit total')
    ax[1].plot(t_vdf, np.linalg.norm(v_bulk_bf*1e-3, axis=1), '.', color = 'black', label = 'moments')
    ax[1].legend()
    ax[1].set_ylabel('$V_{mag} (km/s)$')
    ax[1].set_xticks([])

    ax[2].plot(t_vdf, T_mag_c, color = 'teal', label = 'core')
    ax[2].plot(t_vdf, T_mag_b, color = '#CC5500', label = 'beam')
    ax[2].plot(t_vdf, total_T, color = 'darkred', label = 'fit total')
    ax[2].plot(t_vdf, T, '.', color = 'black', label = 'moments')
    ax[2].legend()
    ax[2].set_ylabel('$T (eV)$')
    # ax[2].set_ylim([0, 40])
    ax[2].xaxis.set_major_locator(locator)
    ax[2].xaxis.set_major_formatter(formatter)

def plot_combined_temperatures(t_vdf, T_par, T_perp, T_par_c, T_perp_c, T_par_b, T_perp_b, total_T_par, total_T_perp):
    """
    Plot combined paralle and perpendicular temperatures of core and beam populations.
    """

    fig, ax = plt.subplots(nrows = 2, figsize = [30, 20])

    ax[0].plot(t_vdf, T_par_c, color = 'teal', label = 'core')
    ax[0].plot(t_vdf, T_par_b, color = '#CC5500', label = 'beam')
    ax[0].plot(t_vdf, total_T_par, color = 'darkred', label = 'fit total')
    ax[0].plot(t_vdf, T_par, '.', color = 'black', label = 'moments')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax[0].legend()
    ax[0].set_ylabel('$T_{\parallel} (eV)$')
    ax[0].set_xticks([])

    ax[1].plot(t_vdf, T_perp_c, color = 'teal', label = 'core')
    ax[1].plot(t_vdf, T_perp_b, color = '#CC5500', label = 'beam')
    ax[1].plot(t_vdf, total_T_perp, color = 'darkred', label = 'fit total')
    ax[1].plot(t_vdf, T_perp, '.', color = 'black', label = 'moments')
    ax[1].legend()
    ax[1].set_ylabel('$T_{\perp} (eV)$')
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

def plot_core_beam_ratios(t_vdf, v_drift_par, VA, n_ratio, T_ratio):

    fig, ax = plt.subplots(nrows = 3, figsize = [30, 20])

    ax[0].plot(t_vdf, v_drift_par, color = 'teal', label = '$v_{bc}$')
    ax[0].plot(t_vdf, VA, color = 'darkred', label = '$v_A$')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax[0].legend()
    ax[0].set_ylabel('Parallel V drift')
    ax[0].set_xticks([])

    ax[1].plot(t_vdf, n_ratio, color = 'darkred', label = '$n_b / n_c$')
    # ax[1].plot(t_vdf:, nb_all:*1e-6 / n:, color = 'darkred', label = '$n_b / n_c$')
    ax[1].legend()
    ax[1].set_ylabel('Density ratio ($n_b / n_c$)')
    ax[1].set_xticks([])

    ax[2].plot(t_vdf, T_ratio, color = 'darkred', label = '$T_b / T_c$')
    ax[2].legend()
    ax[2].set_ylabel('Temperature ratio ($T_b / T_c$)')
    ax[2].set_xticks([])
    ax[2].xaxis.set_major_locator(locator)
    ax[2].xaxis.set_major_formatter(formatter)

def plot_instability_thresholds(t_vdf, beta_par, a_p):

    x = np.logspace(-1, 1, 1000)
    y = np.logspace(-1, 1, 1000)

    # Values from Verscharen et al "The multi-scale nature of the solar wind" (2019)

    y_mirror = 1 + (1.04 / ((x - (-0.012))**(0.633)))
    y_alf_fir = 1 + (-1.447 / ((x - (-0.148))**(1)))
    y_par_fir = 1 + (-0.647 / ((x - (0.713))**(0.583)))
    y_cyc = 1 + (0.649 / ((x - (-0))**(0.649)))

    fig, ax = plt.subplots(nrows=2, figsize = [40, 30])

    "Evaluate a 2D density (PDF) using gaussian_kde"
    xy = np.vstack([beta_par, a_p])
    kde = gaussian_kde(xy)
    z = kde(xy) # This gives the density at each point

    c = ax[0].scatter(beta_par, a_p, c = z, cmap=cmocean.cm.thermal)
    # hist = ax[0].hist2d(beta_par, a_p, cmap = cmocean.cm.gray_r, bins=50)
    # c = ax.scatter(beta_par, a_p, c = z, cmap=cmocean.cm.thermal,  norm=LogNorm())
    ax[0].plot(x, y_mirror, '-.', color = 'red', label = 'Mirror', lw=3)
    ax[0].plot(x, y_alf_fir, '-.', color = 'orange', label = 'Alfven Firehose', lw=3)
    ax[0].plot(x, y_par_fir, '-.', color = 'blue', label = 'Parallel Firehose', lw=3)
    ax[0].plot(x, y_cyc, '--', color = 'black', label = 'Proton Cyclotron', lw=3)
    ax[0].hlines(1, x[0], x[-1], ls = '--', color = 'grey')

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim([x[0], x[-1]])
    ax[0].set_ylim([y[0], y[-1]])
    ax[0].set_xlabel('$\\beta_{\parallel, p}$')
    ax[0].set_ylabel('$T_\perp / T_\parallel$')
    ax[0].legend()
    plt.colorbar(c)
    # plt.colorbar(hist[3], ax=ax[0], label='Counts', orientation='vertical')

    "Instability threshold time series" 
    a, b, b_0 = 0.43, 0.42, -0.0004
    PCI = a/((beta_par-b_0)**b)

    a, b, b_0 = 0.77, 0.76, -0.016
    MI = a/((beta_par-b_0)**b)

    a, b, b_0 = -0.47, 0.53, 0.59
    PFH = a/((beta_par-b_0)**b)

    a, b, b_0 = -1.4, 1.0, -0.11
    OFH = a/((beta_par-b_0)**b)

    ax[1].plot(t_vdf, PCI, label = 'PCI', color = "#ab02ff")
    ax[1].plot(t_vdf, MI, label = 'MI', color = "#0206ff")
    ax[1].plot(t_vdf, PFH, label = 'PFH', color = "#168200")
    ax[1].plot(t_vdf, OFH, label = 'OFH', color = "#a70000")
    ax[1].plot(t_vdf,-1 + a_p, label = '$-1 + T_\perp / T_\parallel$', color = "#000000")
    ax[1].set_ylabel('Instability thresholds')
    # ax[1].set_xlim(datetime(2022, 2, 28, 14, 0), datetime(2022, 2, 28, 16, 0))
    ax[1].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    ax[1].set_ylim([-5, 5])

def constraints_plot(t_vdf, constraints_min, constraints_max, in_conds, fitted_params, nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, n, constraint_flag_1d):

    fig, ax = plt.subplots(nrows = 6, ncols=2, figsize = [60, 40])

    ax[0, 0].plot(t_vdf, constraints_min[:, 0] / (n*1e6), linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[0, 0].plot(t_vdf, constraints_max[:, 0] / (n*1e6), linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[0, 0].plot(t_vdf, nc_all / (n*1e6), 'x', label = '$n_c$', color = 'black')
    ax[0, 0].plot(t_vdf[constraint_flag_1d], fitted_params[:, 0][constraint_flag_1d] / (n*1e6)[constraint_flag_1d], 'x', label = '$n_c$', color = 'red')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    # ax[0, 0].legend()
    ax[0, 0].set_ylabel('$N_c$')
    ax[0, 0].set_xticks([])

    ax[1, 0].plot(t_vdf, (constraints_min[:, 1] - in_conds[:, 1]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[1, 0].plot(t_vdf, (constraints_max[:, 1] - in_conds[:, 1]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[1, 0].plot(t_vdf, (vc_all[:, 0] - in_conds[:, 1]) * 1e-3, 'x', label = '$vx_c$', color = 'black')
    ax[1, 0].plot(t_vdf[constraint_flag_1d], (fitted_params[:, 1][constraint_flag_1d] - in_conds[:, 1][constraint_flag_1d]) * 1e-3, 'x', color = 'red')
    # ax[1, 0].legend()
    ax[1, 0].set_ylabel('$V_{x, c}$')
    ax[1, 0].set_xticks([])

    ax[2, 0].plot(t_vdf, (constraints_min[:, 2] - in_conds[:, 2]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[2, 0].plot(t_vdf, (constraints_max[:, 2] - in_conds[:, 2]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[2, 0].plot(t_vdf, (vc_all[:, 1] - in_conds[:, 2]) * 1e-3, 'x', label = '$vy_c$', color = 'black')
    ax[2, 0].plot(t_vdf[constraint_flag_1d], (fitted_params[:, 2][constraint_flag_1d] - in_conds[:, 2][constraint_flag_1d]) * 1e-3, 'x', color = 'red')
    # ax[2, 0].legend()
    ax[2, 0].set_ylabel('$V_{y, c}$')
    ax[2, 0].set_xticks([])

    ax[3, 0].plot(t_vdf, (constraints_min[:, 3] - in_conds[:, 3]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[3, 0].plot(t_vdf, (constraints_max[:, 3] - in_conds[:, 3]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[3, 0].plot(t_vdf, (vc_all[:, 2] - in_conds[:, 3]) * 1e-3, 'x', label = '$vz_c$', color = 'black')
    ax[3, 0].plot(t_vdf[constraint_flag_1d], (fitted_params[:, 3][constraint_flag_1d] - in_conds[:, 3][constraint_flag_1d]) * 1e-3, 'x', color = 'red')
    # ax[3, 0].legend()
    ax[3, 0].set_ylabel('$V_{z, c}$')
    ax[3, 0].set_xticks([])

    ax[4, 0].plot(t_vdf, constraints_min[:, 4] / in_conds[:, 4], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[4, 0].plot(t_vdf, constraints_max[:, 4] / in_conds[:, 4], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[4, 0].plot(t_vdf, vth_par_c_all / in_conds[:, 4], 'x', label = '$vth_{\parallel, c}$', color = 'black')
    ax[4, 0].plot(t_vdf[constraint_flag_1d], fitted_params[:, 4][constraint_flag_1d] / in_conds[:, 4][constraint_flag_1d], 'x', color = 'red')
    # ax[4, 0].legend()
    ax[4, 0].set_ylabel('$V_{th, \parallel, c}$')
    ax[4, 0].set_xticks([])

    ax[5, 0].plot(t_vdf, constraints_min[:, 5] / in_conds[:, 5], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[5, 0].plot(t_vdf, constraints_max[:, 5] / in_conds[:, 5], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[5, 0].plot(t_vdf, vth_perp_c_all / in_conds[:, 5], 'x', label = '$vth_{\perp, c}$', color = 'black')
    ax[5, 0].plot(t_vdf[constraint_flag_1d], fitted_params[:, 5][constraint_flag_1d] / in_conds[:, 5][constraint_flag_1d], 'x', color = 'red')
    # ax[5, 0].legend()
    ax[5, 0].set_ylabel('$V_{th, \perp, c}$')
    ax[5, 0].set_xticks([])
    ax[5, 0].xaxis.set_major_locator(locator)
    ax[5, 0].xaxis.set_major_formatter(formatter)

    ax[0, 1].plot(t_vdf, constraints_min[:, 6] / (n*1e6), linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[0, 1].plot(t_vdf, constraints_max[:, 6] / (n*1e6), linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[0, 1].plot(t_vdf, nb_all / (n*1e6), 'x', label = '$n_b$', color = 'black')
    ax[0, 1].plot(t_vdf[constraint_flag_1d], fitted_params[:, 6][constraint_flag_1d] / (n*1e6)[constraint_flag_1d], 'x', color = 'red')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    # ax[0, 1].legend()
    ax[0, 1].set_ylabel('$N_b$')
    ax[0, 1].set_xticks([])

    ax[1, 1].plot(t_vdf, (constraints_min[:, 7] - in_conds[:, 7]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[1, 1].plot(t_vdf, (constraints_max[:, 7] - in_conds[:, 7]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[1, 1].plot(t_vdf, (vb_all[:, 0] - in_conds[:, 7]) * 1e-3, 'x', label = '$vx_b$', color = 'black')
    ax[1, 1].plot(t_vdf[constraint_flag_1d], (fitted_params[:, 7][constraint_flag_1d] - in_conds[:, 7][constraint_flag_1d]) * 1e-3, 'x', color = 'red')
    # ax[1, 1].legend()
    ax[1, 1].set_ylabel('$V_{x, b}$')
    ax[1, 1].set_xticks([])

    ax[2, 1].plot(t_vdf, (constraints_min[:, 8] - in_conds[:, 8]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[2, 1].plot(t_vdf, (constraints_max[:, 8] - in_conds[:, 8]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[2, 1].plot(t_vdf, (vb_all[:, 1] - in_conds[:, 8]) * 1e-3, 'x', color = 'black')
    ax[2, 1].plot(t_vdf[constraint_flag_1d], (fitted_params[:, 8][constraint_flag_1d] - in_conds[:, 8][constraint_flag_1d]) * 1e-3, 'x', color = 'red')
    # ax[2, 1].legend()
    ax[2, 1].set_ylabel('$V_{y, b}$')
    ax[2, 1].set_xticks([])

    ax[3, 1].plot(t_vdf, (constraints_min[:, 9] - in_conds[:, 9]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[3, 1].plot(t_vdf, (constraints_max[:, 9] - in_conds[:, 9]) * 1e-3, linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[3, 1].plot(t_vdf, (vb_all[:, 2] - in_conds[:, 9]) * 1e-3, 'x', label = '$vz_b$', color = 'black')
    ax[3, 1].plot(t_vdf[constraint_flag_1d], (fitted_params[:, 9][constraint_flag_1d] - in_conds[:, 9][constraint_flag_1d]) * 1e-3, 'x', color = 'red')
    # ax[3, 1].legend()
    ax[3, 1].set_ylabel('$V_{z, b}$')
    ax[3, 1].set_xticks([])

    ax[4, 1].plot(t_vdf, constraints_min[:, 10] / in_conds[:, 10], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[4, 1].plot(t_vdf, constraints_max[:, 10] / in_conds[:, 10], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[4, 1].plot(t_vdf, vth_par_b_all / in_conds[:, 10], 'x', label = '$vth_{\parallel, b}$', color = 'black')
    ax[4, 1].plot(t_vdf[constraint_flag_1d], fitted_params[:, 10][constraint_flag_1d] / in_conds[:, 10][constraint_flag_1d], 'x', color = 'red')
    # ax[4, 1].legend()
    ax[4, 1].set_ylabel('$V_{th, \parallel, b}$')
    ax[4, 1].set_xticks([])

    ax[5, 1].plot(t_vdf, constraints_min[:, 11] / in_conds[:, 11], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[5, 1].plot(t_vdf, constraints_max[:, 11] / in_conds[:, 11], linestyle = 'dashed', color = '#7d005f', lw = 3)
    ax[5, 1].plot(t_vdf, vth_perp_b_all / in_conds[:, 11], 'x', label = '$vth_{\perp, b}$', color = 'black')
    ax[5, 1].plot(t_vdf[constraint_flag_1d], fitted_params[:, 11][constraint_flag_1d] / in_conds[:, 11][constraint_flag_1d], 'x', color = 'red')
    # ax[5, 1].legend()
    ax[5, 1].set_ylabel('$V_{th, \perp, b}$')
    ax[5, 1].set_xticks([])
    ax[5, 1].xaxis.set_major_locator(locator)
    ax[5, 1].xaxis.set_major_formatter(formatter)

def plot_fits_kappa(plot_indices, parameters, vdf_in, errors_in, vx_bf, vy_bf, vz_bf, date1_str, t_vdf, theta, phi, energy, folder, lims=[-10., -6.5], plot_3D = False, save = False):
    
    nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all = parameters
    "PLOT FITS"
    p_ran = 5  # Plotting range parameter

    vels = E_to_v(energy)*1e-3  # Convert energy to velocity

    errors = errors_in

    # Create plot directory if not exist

    folder_path = f'Plots/Poisson_fits_data/{date1_str[0]}_{date1_str[1]}_{date1_str[2]}/kappa/{folder}/Example_fits'
    os.makedirs(folder_path, exist_ok=True)

    for ind in tqdm(plot_indices):
        data_f = vdf_in[ind]
        errors_f = errors[ind]
        # theta_slice = theta_all[ind]
        
        (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c,
        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b) = extract_all_fit_parameters_kappa(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all, kappa_all,
                                                                                                            nb_all, vb_all, vth_par_b_all, vth_perp_b_all, show = False)
        vc_mag = np.linalg.norm([vx_fit_c, vy_fit_c, vz_fit_c]) * 1e-3

        # Model VDFs
        vdf_fit = bi_kappa_bi_max(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c, n_fit_b,
                                vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)
        core_fit = bi_kappa(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, kappa_c)
        beam_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind],
                        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b)
        
        # az_range, el_range = get_index_ranges(vdf, p_ran)

        # Prepare velocity data for integration
        data_integrated = integrate_vdf_over_angles(data_f, theta)
        fit_integrated = integrate_vdf_over_angles(vdf_fit, theta)
        core_integrated = integrate_vdf_over_angles(core_fit, theta)
        beam_integrated = integrate_vdf_over_angles(beam_fit, theta)
        errors_integrated = integrate_vdf_over_angles(errors_f, theta, errors=True)
        
        # Plotting
        fig, ax = plt.subplots(ncols=2, figsize=(40, 15))

        plot_integrated_vdf(
            ax, vels, core_integrated, beam_integrated, data_integrated, 
            fit_integrated, errors_integrated, lims = lims, vc_mag=vc_mag, t_vdf_ind=t_vdf[ind]
            )
        
        # Save figure
        if save == True:
            filename = f'1d_{ind}.png'
            save_figure(fig, folder_path, filename)

        else:
            plt.show()
        
        if plot_3D == True:

            fig_lin, fig_log = plot_energy_grid(
            energy, data_f, vdf_fit, core_fit, beam_fit,
            errors_f, ind, theta, phi, lims=lims, p_ran=p_ran
            )
            
            # Save figure
            if save == True:
                filename_log = f'grid_log_{ind}.png'
                filename_lin = f'grid_{ind}.png'
                save_figure(fig_log, folder_path, filename_log)
                save_figure(fig_lin, folder_path, filename_lin)

            else:
                plt.show(fig_lin)
                plt.show(fig_log)

def plot_fits_kappa_beam(plot_indices, parameters, vdf_in, errors_in, vx_bf, vy_bf, vz_bf, date1_str, t_vdf, theta, phi, energy, folder, lims=[-10., -6.5], plot_3D = False, save = False):
    
    nc_all, vc_all, vth_par_c_all, vth_perp_c_all, nb_all, vb_all, vth_par_b_all, vth_perp_b_all, kappa_all = parameters
    "PLOT FITS"
    p_ran = 5  # Plotting range parameter

    vels = E_to_v(energy)*1e-3  # Convert energy to velocity

    errors = errors_in

    # Create plot directory if not exist

    folder_path = f'Plots/Poisson_fits_data/{date1_str[0]}_{date1_str[1]}_{date1_str[2]}/kappa/{folder}/Example_fits'
    os.makedirs(folder_path, exist_ok=True)

    for ind in tqdm(plot_indices):
        data_f = vdf_in[ind]
        errors_f = errors[ind]
        # theta_slice = theta_all[ind]
        
        (n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c,
        n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b, kappa_b) = extract_all_fit_parameters_kappa_beam(ind, nc_all, vc_all, vth_par_c_all, vth_perp_c_all,
                                                                                                            nb_all, vb_all, vth_par_b_all, vth_perp_b_all, kappa_all, show = False)
        vc_mag = np.linalg.norm([vx_fit_c, vy_fit_c, vz_fit_c]) * 1e-3

        # Model VDFs
        vdf_fit = bi_max_bi_kappa(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c, n_fit_b,
                                vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b, kappa_b)
        core_fit = bi_Max(vx_bf[ind], vy_bf[ind], vz_bf[ind],
                        n_fit_c, vx_fit_c, vy_fit_c, vz_fit_c, v_th_par_fit_c, v_th_perp_fit_c)
        beam_fit = bi_kappa(vx_bf[ind], vy_bf[ind], vz_bf[ind], n_fit_b, vx_fit_b, vy_fit_b, vz_fit_b, v_th_par_fit_b, v_th_perp_fit_b, kappa_b)
        
        # az_range, el_range = get_index_ranges(vdf, p_ran)

        # Prepare velocity data for integration
        data_integrated = integrate_vdf_over_angles(data_f, theta)
        fit_integrated = integrate_vdf_over_angles(vdf_fit, theta)
        core_integrated = integrate_vdf_over_angles(core_fit, theta)
        beam_integrated = integrate_vdf_over_angles(beam_fit, theta)
        errors_integrated = integrate_vdf_over_angles(errors_f, theta, errors=True)
        
        # Plotting
        fig, ax = plt.subplots(ncols=2, figsize=(40, 15))

        plot_integrated_vdf(
            ax, vels, core_integrated, beam_integrated, data_integrated, 
            fit_integrated, errors_integrated, lims = lims, vc_mag=vc_mag, t_vdf_ind=t_vdf[ind]
            )
        
        # Save figure
        if save == True:
            filename = f'1d_{ind}.png'
            save_figure(fig, folder_path, filename)

        else:
            plt.show()
        
        if plot_3D == True:

            fig_lin, fig_log = plot_energy_grid(
            energy, data_f, vdf_fit, core_fit, beam_fit,
            errors_f, ind, theta, phi, lims=lims, p_ran=p_ran
            )
            
            # Save figure
            if save == True:
                filename_log = f'grid_log_{ind}.png'
                filename_lin = f'grid_{ind}.png'
                save_figure(fig_log, folder_path, filename_log)
                save_figure(fig_lin, folder_path, filename_lin)

            else:
                plt.show(fig_lin)
                plt.show(fig_log)

def plot_comparison_time_series(t_vdf, total_n_p, v_p, total_T_p, t_h, n_h, v_h, T_h, n, v_bulk_mag, T):

    fig, ax = plt.subplots(nrows = 3, figsize = [30, 20])

    ax[0].plot(t_vdf, total_n_p, color = 'darkred', label = 'Poisson')
    ax[0].plot(t_h, n_h, color = '#CC5500', label = 'GMM')
    ax[0].plot(t_vdf, n, '.', color = 'black', label = 'moments')
    ax[0].legend(loc = 4)
    ax[0].set_ylabel('$n (cm^{-3})$')
    ax[0].set_xticks([])
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    # ax[0].xaxis.set_major_locator(locator)
    # ax[0].xaxis.set_major_formatter(formatter)

    ax[0].set_title('Density comparison')

    ax[1].plot(t_vdf, v_p, color = 'darkred', label = 'Poisson')
    ax[1].plot(t_h, v_h, color = '#CC5500', label = 'GMM')
    ax[1].plot(t_vdf, v_bulk_mag, '.', color = 'black', label = 'moments')
    ax[1].legend(loc = 4)
    ax[1].set_ylabel('$v (km/s)$')
    ax[1].set_xticks([])

    ax[1].set_title('Bulk speed mag comparison')

    ax[2].plot(t_vdf, total_T_p, color = 'darkred', label = 'Poisson')
    ax[2].plot(t_h, T_h, color = '#CC5500', label = 'GMM')
    ax[2].plot(t_vdf, T, '.', color = 'black', label = 'moments')
    ax[2].legend(loc = 4)
    ax[2].set_ylabel('$T (eV)$')
    ax[2].set_xticks([])
    ax[2].xaxis.set_major_locator(locator)
    ax[2].xaxis.set_major_formatter(formatter)

    ax[2].set_title('Temperature comparison')

def plot_T_comparison(t_vdf, total_T_par_p, total_T_perp_p, t_h, Tpar_h, Tperp_h, T_par, T_perp):

    fig, ax = plt.subplots(nrows = 2, figsize = [30, 20])

    ax[0].plot(t_vdf, total_T_par_p, color = 'darkred', label = 'Poisson')
    ax[0].plot(t_h, Tpar_h, color = '#CC5500', label = 'GMM')
    ax[0].plot(t_vdf, T_par, '.', color = 'black', label = 'moments')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax[0].legend(loc = 4)
    ax[0].set_ylabel('$T_\parallel (eV)$')
    ax[0].set_xticks([])

    ax[1].plot(t_vdf, total_T_perp_p, color = 'darkred', label = 'Poisson')
    ax[1].plot(t_h, Tperp_h, color = '#CC5500', label = 'GMM')
    ax[1].plot(t_vdf, T_perp, '.', color = 'black', label = 'moments')

    ax[1].legend(loc = 4)
    ax[1].set_ylabel('$T_\perp (eV)$')
    ax[1].set_xticks([])

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)

def plot_residual_comparison(t_vdf, total_n_p, v_p, total_T_p, t_h, n_h, v_h, T_h):

    t_vdf_s = np.array([dt.replace(microsecond=0) for dt in t_vdf])
    common_times = np.intersect1d(t_vdf_s, t_h)

    ind_p = np.where(np.isin(t_vdf_s, common_times))[0]
    ind_h = np.where(np.isin(t_h, common_times))[0]

    "Histogram of residuals between Poisson Fit and GMM"
    fig, ax = plt.subplots(nrows = 3, figsize = [20, 30])

    ax[0].hist(total_n_p[ind_p] - n_h[ind_h], bins = 20, color = 'darkred')
    ax[0].set_xlabel('$n_P$ - $n_{GMM}$ ($cm^{-3}$)')
    ax[0].set_ylabel('Occurences')
    ax[0].text(.99, .99, '$\hat{n}_P$'+f' = {np.mean(total_n_p):.1f}'+'$cm^{-3}$', ha='right', va='top', transform=ax[0].transAxes, fontsize = 50)

    ax[1].hist(v_p[ind_p] - v_h[ind_h], bins = 20, color = 'darkred')
    ax[1].set_xlabel('$V_P$ - $V_{GMM}$ ($km/s$)')
    ax[1].set_ylabel('Occurences')
    ax[1].text(.99, .99, '$\hat{V}_P$ = '+f'{np.mean(v_p):.0f}'+'km/s', ha='right', va='top', transform=ax[1].transAxes, fontsize = 50)

    ax[2].hist(total_T_p[ind_p] - T_h[ind_h], bins = 20, color = 'darkred')
    ax[2].set_xlabel('$T_P$ - $T_{GMM}$ ($eV$)')
    ax[2].set_ylabel('Occurences')
    ax[2].text(.99, .99, '$\hat{T}_P$ = '+f'{np.mean(total_T_p):.1f}'+'eV', ha='right', va='top', transform=ax[2].transAxes, fontsize = 50)

    plt.show()

    "Time series of residuals"
    "Histogram of residuals between Poisson Fit and GMM"
    fig, ax = plt.subplots(nrows = 3, figsize = [20, 30])

    ax[0].plot(t_vdf[ind_p], total_n_p[ind_p] - n_h[ind_h], color = 'darkred')
    ax[0].set_ylabel('$n_P$ - $n_{GMM}$ ($cm^{-3}$)')
    # ax[0].set_ylabel('Occurences')
    ax[0].text(.99, .99, '$\hat{n}_P$'+f' = {np.mean(total_n_p):.1f}'+'$cm^{-3}$', ha='right', va='top', transform=ax[0].transAxes, fontsize = 50)

    ax[1].plot(t_vdf[ind_p], v_p[ind_p] - v_h[ind_h], color = 'darkred')
    ax[1].set_ylabel('$V_P$ - $V_{GMM}$ ($km/s$)')
    # ax[1].set_ylabel('Occurences')
    ax[1].text(.99, .99, '$\hat{V}_P$ = '+f'{np.mean(v_p):.0f}'+'km/s', ha='right', va='top', transform=ax[1].transAxes, fontsize = 50)

    ax[2].plot(t_vdf[ind_p], total_T_p[ind_p] - T_h[ind_h], color = 'darkred')
    ax[2].set_ylabel('$T_P$ - $T_{GMM}$ ($eV$)')
    # ax[2].set_ylabel('Occurences')
    ax[2].text(.99, .99, '$\hat{T}_P$ = '+f'{np.mean(total_T_p):.1f}'+'eV', ha='right', va='top', transform=ax[2].transAxes, fontsize = 50)

    plt.show()