
from importlib import reload
import numpy as np
from scipy.stats import poisson, kstest

from Poisson_fit_functions_2 import *
from gen_funcs_2 import *

def model_lambda_params(params, design):
    """
    Example: a model that maps params to lambda_i for each observation.
    Replace this with your actual function g(design_i, params) that
    returns nonnegative means for each i.
    Example here: lambda_i = exp( beta0 + beta1 * design_i )
    """

    n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1, n_var2, v_x2, v_th_par2, v_th_perp2 = params 
    v_y2, v_z2 = v_y1, v_z1

    ux, uy, uz, G, dt = design

    counts = counts_double_biMaxwellian(ux, uy, uz, n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1, n_var2, v_x2, v_y2, v_z2, v_th_par2, v_th_perp2, G, dt)
    return counts

def model_lambda_params_core_only(params, design):
    """
    Example: a model that maps params to lambda_i for each observation.
    Replace this with your actual function g(design_i, params) that
    returns nonnegative means for each i.
    Example here: lambda_i = exp( beta0 + beta1 * design_i )
    """

    n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1 = params 

    ux, uy, uz, G, dt = design

    counts = counts_biMaxwellian(ux, uy, uz, n_var1, v_x1, v_y1, v_z1, v_th_par1, v_th_perp1, G, dt)
    return counts

def randomized_pit(xs, lambdas, rng):
    xs = np.asarray(xs).ravel()
    lambdas = np.asarray(lambdas).ravel()
    v = rng.random(xs.shape[0])
    x_minus_1 = np.round(xs) - 1
    F_minus = poisson.cdf(x_minus_1, lambdas)
    F_minus[x_minus_1 < 0] = 0.0
    F_x = poisson.cdf(np.round(xs), lambdas)
    U = F_minus + v * (F_x - F_minus)
    return U

def ks_gof(xs, design, parameters, seed=123):
    """
    Apply KS goodness-of-fit test with bootstrap to account for parameter estimation.

    Parameters
    ----------
    xs : array-like
        Observed counts.
    seed : int
        RNG seed.

    Returns
    -------
    'ks_obs'  : observed KS statistic
    'p_value' : p-value
    """
    rng = np.random.default_rng(seed)

    # 1. Get fitted lambdas under observed data
    lambdas_hat = model_lambda_params(parameters, design)

    mask_obs = np.isfinite(xs) & np.isfinite(lambdas_hat) & (lambdas_hat > 0)
    xs_filtered = xs[mask_obs]
    lambdas_hat_filtered = lambdas_hat[mask_obs]

    U_obs = randomized_pit(xs_filtered, lambdas_hat_filtered, rng)
    k_test = kstest(U_obs, "uniform")
    ks_obs = k_test.statistic
    p_obs = k_test.pvalue

    return  np.array([ks_obs, p_obs])

def ks_gof_core_only(xs, design, parameters, seed=123):
    """
    Apply KS goodness-of-fit test with bootstrap to account for parameter estimation.

    Parameters
    ----------
    xs : array-like
        Observed counts.
    seed : int
        RNG seed.

    Returns
    -------
    'ks_obs'  : observed KS statistic
    'p_value' : p-value
    """
    rng = np.random.default_rng(seed)

    # 1. Get fitted lambdas under observed data
    lambdas_hat = model_lambda_params_core_only(parameters, design)

    mask_obs = np.isfinite(xs) & np.isfinite(lambdas_hat) & (lambdas_hat > 0)
    xs_filtered = xs[mask_obs]
    lambdas_hat_filtered = lambdas_hat[mask_obs]

    U_obs = randomized_pit(xs_filtered, lambdas_hat_filtered, rng)
    k_test = kstest(U_obs, "uniform")
    ks_obs = k_test.statistic
    p_obs = k_test.pvalue

    return  np.array([ks_obs, p_obs])

def dev_gof(xs, design, parameters):
    """
    Compute Poisson deviance and approximate chi-square p-value.
    
    Parameters
    ----------
    xs : array-like
        Observed counts (flattened).
    lambdas : array-like
        Fitted means corresponding to each observation.
    n_params : int
        Number of fitted parameters in the model.
    
    Returns
    -------
    D : float
        Deviance statistic.
    df : int
        Degrees of freedom (n - p).
    p_value : float
        Chi-square p-value.
    """

    lambdas = model_lambda_params(parameters, design)
    n_params = len(parameters)
    xs = np.asarray(xs).ravel()
    lambdas = np.asarray(lambdas).ravel()
    mask = np.isfinite(xs) & np.isfinite(lambdas) & (lambdas > 0)
    xs, lambdas = xs[mask], lambdas[mask]

    # Deviance
    L_data, L_array = poisson_log_likelihood4(xs, lambdas)
    L_sat =  poisson_log_likelihood3(xs, xs)
    # print(L_sat, L_data)

    D_obs = 2 * (L_sat - L_data)

    df = len(xs) - n_params

    D_norm = D_obs / df
    L_mean = np.mean(L_array)
    L_std = np.std(L_array)

    return (D_norm, df, L_mean, L_std)

def dev_gof_core_only(xs, design, parameters):
    """
    Compute Poisson deviance and approximate chi-square p-value.
    
    Parameters
    ----------
    xs : array-like
        Observed counts (flattened).
    lambdas : array-like
        Fitted means corresponding to each observation.
    n_params : int
        Number of fitted parameters in the model.
    
    Returns
    -------
    D : float
        Deviance statistic.
    df : int
        Degrees of freedom (n - p).
    p_value : float
        Chi-square p-value.
    """

    lambdas = model_lambda_params_core_only(parameters, design)
    n_params = len(parameters)
    xs = np.asarray(xs).ravel()
    lambdas = np.asarray(lambdas).ravel()
    mask = np.isfinite(xs) & np.isfinite(lambdas) & (lambdas > 0)
    xs, lambdas = xs[mask], lambdas[mask]

    # Deviance
    L_data, L_array = poisson_log_likelihood4(xs, lambdas)
    L_sat =  poisson_log_likelihood3(xs, xs)
    # print(L_sat, L_data)

    D_obs = 2 * (L_sat - L_data)

    df = len(xs) - n_params

    D_norm = D_obs / df
    L_mean = np.mean(L_array)
    L_std = np.std(L_array)

    return (D_norm, df, L_mean, L_std)