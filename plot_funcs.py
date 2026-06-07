"""
Author: Charalambos Ioannou
Institution: UCL / MSSL
Email: charalambos.ioannou.22@ucl.ac.uk
GitHub: @Cioannou101
Created: 2026-06-07

This script contains functions for plotting VDFs, fits, and related components.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from gen_funcs import *
from Poisson_fit_functions import double_bi_Max, bi_Max


def save_figure(fig, folder_path, filename):
    save_path = os.path.join(folder_path, filename)
    fig.savefig(save_path)
    plt.close(fig)

def plot_integrated_vdf(ax, vels, core_f, beam_f, data_f, fit_f, errors_f, vc_mag, t_vdf_ind, lims = [1e-10, 1e-6], core_only = False):

    if core_only == True:
    
        ax[1].plot(vels, np.log10(core_f), '--', color='teal', label='core', lw=3)
        ax[1].errorbar(vels, np.log10(data_f), yerr=errors_f/(data_f * np.log(10)), fmt='x', ms=9, mew=2, color='steelblue', label='data')
        # ax[1].errorbar(vels, np.log10(data_f), fmt='x', ms=9, mew=2, color='steelblue', label='data')
        ax[1].set_ylim(bottom=-10, top=-6.5)
        ax[1].set_xlabel("V (km/s)")
        ax[1].set_ylabel("VDF")
        ax[1].legend(loc=1, frameon=False)
        ax[1].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[1].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')
    
        ax[0].plot(vels, core_f, '--', color='teal', label='core', lw=3)
        ax[0].errorbar(vels, data_f, yerr=errors_f, fmt='x', ms=9, mew=2, color='steelblue', label='data')
        
        ax[0].set_xlabel("V (km/s)")
        ax[0].set_ylabel("VDF")
        ax[0].legend(loc=1, frameon=False)
        ax[0].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[0].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')
        
    else:
        ax[1].plot(vels, core_f, '--', color='teal', label='core', lw=3)
        ax[1].plot(vels, beam_f, '--', color='#CC5500', label='beam', lw=3)
        ax[1].errorbar(vels, data_f, yerr=errors_f, fmt='x', ms=9, mew=2, color='steelblue', label='data')
        # ax[1].errorbar(vels, np.log10(data_f), fmt='x', ms=9, mew=2, color='steelblue', label='data')
        ax[1].plot(vels, fit_f, color='red', label='fit', lw=3)
        ax[1].set_ylim(bottom=lims[0], top=lims[1])
        ax[1].set_xlabel("V (km/s)")
        ax[1].set_ylabel("VDF")
        ax[1].legend(loc=1, frameon=False)
        ax[1].set_xlim([vc_mag - 200, vc_mag + 300])
        ax[1].set_title(f'Integrated over $\\Theta$ and $\\Phi$ at {t_vdf_ind}')
        ax[1].set_yscale('log')

        ax[0].plot(vels, core_f, '--', color='teal', label='core', lw=3)
        ax[0].plot(vels, beam_f, '--', color='#CC5500', label='beam', lw=3)
        ax[0].errorbar(vels, data_f, yerr=errors_f, fmt='x', ms=9, mew=2, color='steelblue', label='data')
        ax[0].plot(vels, fit_f, color='red', label='fit', lw=3)
        
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
