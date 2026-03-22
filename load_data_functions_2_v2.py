# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:43:18 2025

@author: CI
"""

from hapiclient import hapi, hapitime2datetime
import pandas as pd
import numpy as np
import datetime
import cdflib
import os


def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:~
      DATE - a python datetime object
    """
    new_times = []
    
    for i in range(len(date)):
    
        timestamp = ((date[i] - np.datetime64('1970-01-01T00:00:00'))
                     / np.timedelta64(1, 's'))
        
        new_date = datetime.datetime.utcfromtimestamp(timestamp)
        
        new_times.append(new_date)
        
    return np.array(new_times)


# year, month, day = 2021, 8, 5

# d_range = 1

def load_data_PAS_cdaweb_N(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_SWA-PAS-GRND-MOM'
    parameters = 'N'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop) #this line gets the data
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    """
    the above line separates the time and parameters to their own column
    So params[0] is the time, params[1] is density, params[2] is the 3d V_srf, and params[3] is T
    Then you can do whatever you want with the data. I put them in panda dataframes for convinience. 
    """
    print(paramnames)
    full_paramnames = ['Time', 'N']
    full_params = [params[0], params[1]]
    
    df_n = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_n['Time'] = hapitime2datetime(data['Time'])
    df_n['Time'] = pd.to_datetime(df_n['Time'])
    df_n['Time'] = df_n['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_n[full_paramnames[i]] = pd.to_numeric(df_n[full_paramnames[i]])
    fills = []
    for i in range(len(meta['parameters'])):
            if meta['parameters'][i]['name'] in paramnames:
                    fills.append(meta['parameters'][i]['fill'])
    fills = [float(fill) for fill in fills[1::]]
    
    time = to_datetime(np.array(df_n['Time']))
    
    if return_t == True:
        return time, np.array(df_n['N'])
    
    if return_t == False:
        return np.array(df_n['N'])

# time_n, n = load_data_PAS_cdaweb_N(year, month, day, d_range)

def load_data_PAS_cdaweb_Vsrf(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_SWA-PAS-GRND-MOM'
    parameters = 'V_SRF'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop) #this line gets the data
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    """
    the above line separates the time and parameters to their own column
    So params[0] is the time, params[1] is density, params[2] is the 3d V_srf, and params[3] is T
    Then you can do whatever you want with the data. I put them in panda dataframes for convinience. 
    """
    print(paramnames)
    
    Vx = params[1][:, 0]
    Vy = params[1][:, 1]
    Vz = params[1][:, 2]
    
    Vx[np.isnan(Vx)] = np.nan
    Vy[np.isnan(Vy)] = np.nan
    Vz[np.isnan(Vz)] = np.nan
    
    full_paramnames = ['Time','Vx', 'Vy', 'Vz']
    full_params = [params[0], Vx, Vy, Vz]
    
    df_n = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_n['Time'] = hapitime2datetime(data['Time'])
    df_n['Time'] = pd.to_datetime(df_n['Time'])
    df_n['Time'] = df_n['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_n[full_paramnames[i]] = pd.to_numeric(df_n[full_paramnames[i]])
    fills = []
    for i in range(len(meta['parameters'])):
            if meta['parameters'][i]['name'] in paramnames:
                    fills.append(meta['parameters'][i]['fill'])
    fills = [float(fill) for fill in fills[1::]]
    
    time = to_datetime(np.array(df_n['Time']))
    
    if return_t == True:
        return time, np.array([np.array(df_n['Vx']), np.array(df_n['Vy']), np.array(df_n['Vz'])]).T
    
    if return_t == False:
        return np.array([np.array(df_n['Vx']), np.array(df_n['Vy']), np.array(df_n['Vz'])]).T
        
# t_vsrf, v_srf = load_data_PAS_cdaweb_Vsrf(year, month, day, d_range)

def load_data_PAS_cdaweb_Vrtn(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_SWA-PAS-GRND-MOM'
    parameters = 'V_RTN'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop) #this line gets the data
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    """
    the above line separates the time and parameters to their own column
    So params[0] is the time, params[1] is density, params[2] is the 3d V_srf, and params[3] is T
    Then you can do whatever you want with the data. I put them in panda dataframes for convinience. 
    """
    print(paramnames)
    
    Vx = params[1][:, 0]
    Vy = params[1][:, 1]
    Vz = params[1][:, 2]
    
    Vx[np.isnan(Vx)] = np.nan
    Vy[np.isnan(Vy)] = np.nan
    Vz[np.isnan(Vz)] = np.nan
    
    full_paramnames = ['Time','Vx', 'Vy', 'Vz']
    full_params = [params[0], Vx, Vy, Vz]
    
    df_n = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_n['Time'] = hapitime2datetime(data['Time'])
    df_n['Time'] = pd.to_datetime(df_n['Time'])
    df_n['Time'] = df_n['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_n[full_paramnames[i]] = pd.to_numeric(df_n[full_paramnames[i]])
    fills = []
    for i in range(len(meta['parameters'])):
            if meta['parameters'][i]['name'] in paramnames:
                    fills.append(meta['parameters'][i]['fill'])
    fills = [float(fill) for fill in fills[1::]]
    
    time = to_datetime(np.array(df_n['Time']))
    
    if return_t == True:
        return time, np.array([np.array(df_n['Vx']), np.array(df_n['Vy']), np.array(df_n['Vz'])]).T
    
    if return_t == False:
        return np.array([np.array(df_n['Vx']), np.array(df_n['Vy']), np.array(df_n['Vz'])]).T
        
# t_rtn, v_rtn = load_data_PAS_cdaweb_Vrtn(year, month, day, d_range)

def load_data_PAS_cdaweb_Psrf(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_SWA-PAS-GRND-MOM'
    parameters = 'P_SRF'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop) #this line gets the data
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    """
    the above line separates the time and parameters to their own column
    So params[0] is the time, params[1] is density, params[2] is the 3d V_srf, and params[3] is T
    Then you can do whatever you want with the data. I put them in panda dataframes for convinience. 
    """
    print(paramnames)
    
    P1 = params[1][:, 0]
    P2 = params[1][:, 1]
    P3 = params[1][:, 2]
    P4 = params[1][:, 3]
    P5 = params[1][:, 4]
    P6 = params[1][:, 5]
    
    P1[np.isnan(P1)] = np.nan
    P2[np.isnan(P2)] = np.nan
    P3[np.isnan(P3)] = np.nan
    P4[np.isnan(P4)] = np.nan
    P5[np.isnan(P5)] = np.nan
    P6[np.isnan(P6)] = np.nan
    
    full_paramnames = ['Time', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    full_params = [params[0], P1, P2, P3, P4, P5, P6]
    
    df_n = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_n['Time'] = hapitime2datetime(data['Time'])
    df_n['Time'] = pd.to_datetime(df_n['Time'])
    df_n['Time'] = df_n['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_n[full_paramnames[i]] = pd.to_numeric(df_n[full_paramnames[i]])
    fills = []
    for i in range(len(meta['parameters'])):
            if meta['parameters'][i]['name'] in paramnames:
                    fills.append(meta['parameters'][i]['fill'])
    fills = [float(fill) for fill in fills[1::]]
    
    time = to_datetime(np.array(df_n['Time']))
    
    P_tensor = np.zeros([len(df_n['P1']), 3, 3])

    for i in range(len(df_n['P1'])):
        
        P_tensor[i, 0, 0] = df_n['P1'][i]
        P_tensor[i, 1, 1] = df_n['P2'][i]
        P_tensor[i, 2, 2] = df_n['P3'][i]
        P_tensor[i, 0, 1] = df_n['P4'][i]
        P_tensor[i, 1, 0] = df_n['P4'][i]
        P_tensor[i, 1, 2] = df_n['P6'][i]
        P_tensor[i, 2, 1] = df_n['P6'][i]
        P_tensor[i, 0, 2] = df_n['P5'][i]
        P_tensor[i, 2, 0] = df_n['P5'][i]
    
    if return_t == True:
        return time, P_tensor
    
    if return_t == False:
        return P_tensor

# t_psrf, P_srf = load_data_PAS_cdaweb_Psrf(year, month, day, d_range)

def load_data_PAS_cdaweb_Prtn(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_SWA-PAS-GRND-MOM'
    parameters = 'P_RTN'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop) #this line gets the data
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    """
    the above line separates the time and parameters to their own column
    So params[0] is the time, params[1] is density, params[2] is the 3d V_srf, and params[3] is T
    Then you can do whatever you want with the data. I put them in panda dataframes for convinience. 
    """
    print(paramnames)
    
    P1 = params[1][:, 0]
    P2 = params[1][:, 1]
    P3 = params[1][:, 2]
    P4 = params[1][:, 3]
    P5 = params[1][:, 4]
    P6 = params[1][:, 5]
    
    P1[np.isnan(P1)] = np.nan
    P2[np.isnan(P2)] = np.nan
    P3[np.isnan(P3)] = np.nan
    P4[np.isnan(P4)] = np.nan
    P5[np.isnan(P5)] = np.nan
    P6[np.isnan(P6)] = np.nan
    
    full_paramnames = ['Time', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    full_params = [params[0], P1, P2, P3, P4, P5, P6]
    
    df_n = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_n['Time'] = hapitime2datetime(data['Time'])
    df_n['Time'] = pd.to_datetime(df_n['Time'])
    df_n['Time'] = df_n['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_n[full_paramnames[i]] = pd.to_numeric(df_n[full_paramnames[i]])
    fills = []
    for i in range(len(meta['parameters'])):
            if meta['parameters'][i]['name'] in paramnames:
                    fills.append(meta['parameters'][i]['fill'])
    fills = [float(fill) for fill in fills[1::]]
    
    time = to_datetime(np.array(df_n['Time']))
    
    P_tensor = np.zeros([len(df_n['P1']), 3, 3])

    for i in range(len(df_n['P1'])):
        
        P_tensor[i, 0, 0] = df_n['P1'][i]
        P_tensor[i, 1, 1] = df_n['P2'][i]
        P_tensor[i, 2, 2] = df_n['P3'][i]
        P_tensor[i, 0, 1] = df_n['P4'][i]
        P_tensor[i, 1, 0] = df_n['P4'][i]
        P_tensor[i, 1, 2] = df_n['P6'][i]
        P_tensor[i, 2, 1] = df_n['P6'][i]
        P_tensor[i, 0, 2] = df_n['P5'][i]
        P_tensor[i, 2, 0] = df_n['P5'][i]
    
    if return_t == True:
        return time, P_tensor
    
    if return_t == False:
        return P_tensor
        
# t_prtn, P_rtn = load_data_PAS_cdaweb_Prtn(year, month, day, d_range)

def load_data_PAS_cdaweb_T(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_SWA-PAS-GRND-MOM'
    parameters = 'T'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop) #this line gets the data
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    """
    the above line separates the time and parameters to their own column
    So params[0] is the time, params[1] is density, params[2] is the 3d V_srf, and params[3] is T
    Then you can do whatever you want with the data. I put them in panda dataframes for convinience. 
    """
    print(paramnames)
    full_paramnames = ['Time', 'T']
    full_params = [params[0], params[1]]
    
    df_n = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_n['Time'] = hapitime2datetime(data['Time'])
    df_n['Time'] = pd.to_datetime(df_n['Time'])
    df_n['Time'] = df_n['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_n[full_paramnames[i]] = pd.to_numeric(df_n[full_paramnames[i]])
    fills = []
    for i in range(len(meta['parameters'])):
            if meta['parameters'][i]['name'] in paramnames:
                    fills.append(meta['parameters'][i]['fill'])
    fills = [float(fill) for fill in fills[1::]]
    
    time = to_datetime(np.array(df_n['Time']))
    
    if return_t == True:
        return time, np.array(df_n['T'])
    
    if return_t == False:
        return np.array(df_n['T'])
        
# t_T, T = load_data_PAS_cdaweb_T(year, month, day, d_range)
        
def load_data_PAS_cdaweb_Brtn(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
     
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_MAG-RTN-NORMAL'
    parameters = 'B_RTN'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop)
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    print(paramnames)
    
    Bx = params[1][:, 0]
    By = params[1][:, 1]
    Bz = params[1][:, 2]
    
    Bx[np.isnan(Bx)] = np.nan
    By[np.isnan(Bx)] = np.nan
    Bz[np.isnan(Bx)] = np.nan
    
    B_mag = np.linalg.norm(params[1], axis = 1)
    
    full_paramnames = ['Time', 'Bx', 'By', 'Bz', 'B_mag']
    full_params = [params[0], Bx, By, Bz, B_mag]
    
    df_B = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_B['Time'] = hapitime2datetime(data['Time'])
    df_B['Time'] = pd.to_datetime(df_B['Time'])
    df_B['Time'] = df_B['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_B[full_paramnames[i]] = pd.to_numeric(df_B[full_paramnames[i]], errors = 'coerce')
    
    time = to_datetime(np.array(df_B['Time']))
    
    if return_t == True:
        return time, np.array([np.array(df_B['Bx']), np.array(df_B['By']), np.array(df_B['Bz'])]).T, np.array(df_B['B_mag'])
    
    if return_t == False:
        return np.array([np.array(df_B['Bx']), np.array(df_B['By']), np.array(df_B['Bz'])]).T, np.array(df_B['B_mag'])
        
# t_Brtn, B_rtn, B_rtn_mag = load_data_PAS_cdaweb_Brtn(year, month, day, d_range)

def load_data_PAS_cdaweb_Bsrf(date1, date2, return_t = True):
    "Date is an array containing the year, month, day, hour, minute and second in this order"

    start_datetime = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    end_datetime = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
     
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'SOLO_L2_MAG-SRF-NORMAL'
    parameters = 'B_SRF'
    start      = start_datetime.isoformat()
    stop       = end_datetime.isoformat()
    data, meta = hapi(server, dataset, parameters, start, stop)
    
    paramnames = [meta['parameters'][i]['name'] for i in range(0, len(meta['parameters']))]
    params = [data[paramnames[i]] for i in range(0, len(paramnames))]
    print(paramnames)
    
    Bx = params[1][:, 0]
    By = params[1][:, 1]
    Bz = params[1][:, 2]
    
    Bx[np.isnan(Bx)] = np.nan
    By[np.isnan(Bx)] = np.nan
    Bz[np.isnan(Bx)] = np.nan
    
    B_mag = np.linalg.norm(params[1], axis = 1)
    
    full_paramnames = ['Time', 'Bx', 'By', 'Bz', 'B_mag']
    full_params = [params[0], Bx, By, Bz, B_mag]
    
    df_B = pd.DataFrame(data = np.array(full_params).T, columns = full_paramnames)
    df_B['Time'] = hapitime2datetime(data['Time'])
    df_B['Time'] = pd.to_datetime(df_B['Time'])
    df_B['Time'] = df_B['Time'].apply(lambda t: t.tz_localize(None))
    for i in range(len(full_paramnames)):
            if full_paramnames[i] != 'Time':
                    df_B[full_paramnames[i]] = pd.to_numeric(df_B[full_paramnames[i]], errors = 'coerce')
    
    time = to_datetime(np.array(df_B['Time']))
    
    if return_t == True:
        return time, np.array([np.array(df_B['Bx']), np.array(df_B['By']), np.array(df_B['Bz'])]).T, np.array(df_B['B_mag'])
    
    if return_t == False:
        return np.array([np.array(df_B['Bx']), np.array(df_B['By']), np.array(df_B['Bz'])]).T, np.array(df_B['B_mag'])

# t_Bsrf, B_srf, B_srf_mag = load_data_PAS_cdaweb_Bsrf(year, month, day, d_range)

param_list = ['t_mom', 'N', 'V_srf', 'V_rtn', 'P_srf', 'P_rtn', 'T', 't_B', 'B_rtn', 'B_srf']

def load_data_PAS_cdaweb(param_list, date1, date2):
    
    # check_list = ['N', 'Vsrf', 'Vrtn', 'Psrf', 'Prtn', 'T', 'Brtn', 'Bsrf']

    parameters = [None] * len(param_list)

    for i, param in enumerate(param_list):
        
        if param == 't_mom':
            t_mom = load_data_PAS_cdaweb_N(date1, date2, return_t = True)[0]
            parameters[i] = t_mom
            
        elif param == 'N':
            n = load_data_PAS_cdaweb_N(date1, date2, return_t = False)
            parameters[i] = n
            
        elif param == 'V_srf':
            v_srf = load_data_PAS_cdaweb_Vsrf(date1, date2, return_t = False)
            parameters[i] = v_srf
            
        elif param == 'V_rtn':
            v_rtn = load_data_PAS_cdaweb_Vrtn(date1, date2, return_t = False)
            parameters[i] = v_rtn
            
        elif param == 'P_srf':
            P_srf = load_data_PAS_cdaweb_Psrf(date1, date2, return_t = False)
            parameters[i] = P_srf
            
        elif param == 'P_rtn':
            P_rtn = load_data_PAS_cdaweb_Prtn(date1, date2, return_t = False)
            parameters[i] = P_rtn
            
        elif param == 'T':
            T = load_data_PAS_cdaweb_T(date1, date2, return_t = False)
            parameters[i] = T
            
        elif param == 't_B':
            t_B = load_data_PAS_cdaweb_Bsrf(date1, date2, return_t = True)[0]
            parameters[i] = t_B
            
        elif param == 'B_srf':
            B_srf, B_mag_srf = load_data_PAS_cdaweb_Bsrf(date1, date2, return_t = False)
            parameters[i] = [B_srf, B_mag_srf]
            
        elif param == 'B_rtn':
            B_rtn, B_mag_rtn = load_data_PAS_cdaweb_Brtn(date1, date2, return_t = False)
            parameters[i] = [B_rtn, B_mag_rtn]
            
        else:
            raise Exception(param + 'parameter not in load dataset')
    
    return parameters

# test = load_data_PAS_cdaweb(param_list, year, month, day, d_range)
# t_n, n, v_srf, v_rtn, P_srf, P_rtn, T, t_B, B_srf, B_rtn = test
    
def load_data_PAS_files(fnames, date1, date2, frame = 'SRF', rtn_rot = False):
    "fname contains the name of the file for the VDF, moments and level 1 in that order"

    vdf_cdf = cdflib.CDF(fnames[0])
    moments = cdflib.CDF(fnames[1])
    level1_cdf = cdflib.CDF(fnames[2])
    
    counts = level1_cdf['Counts'] # Counts at each time, azimuth, elevation and energy.
    # time_l1 = level1_cdf['Epoch'] # time in epoch reference for counts.
    t_l1 = cdflib.cdfepoch.to_datetime(level1_cdf['Epoch']) # time in epoch reference for counts.
    
    # time_vdf = vdf_cdf['Epoch'] # time in epoch reference for VDF.
    t_vdf = cdflib.cdfepoch.to_datetime(vdf_cdf['Epoch'])
    phi = vdf_cdf['Azimuth'] # azimuth in degrees.
    theta = vdf_cdf['Elevation'] # elevation in degrees.
    energy = vdf_cdf['Energy'] # energy per charge in eV.
    vdf = vdf_cdf['vdf'] # VDF (time, azimuth, elevation, energy).
    qf = vdf_cdf['quality_factor']
    
    n = moments['N'] # plasma density.
    t_mom = cdflib.cdfepoch.to_datetime(moments['Epoch']) #time in epoch reference for moments.
    T = moments['T'] # scalar temperature in eV.
    P_SRF = moments['P_SRF'] # Pressure is symmetric
    P_RTN = moments['P_RTN']
    # TxTyTz = moments['TxTyTz_SRF']
    V_srf = moments['V_SRF'] # velocities in m/s in spacecraft reference frame
    V_rtn = moments['V_RTN']
    V_solo = moments['V_SOLO_RTN']
    rot_matrix = vdf_cdf['PAS_to_RTN']
    
    t_vdf = to_datetime(np.array(t_vdf))
    
    t_mom = to_datetime(np.array(t_mom))
    # TxTyTz = np.array(TxTyTz)

    # error_vdf = np.array(error_vdf)
    
    t_l1 = to_datetime(np.array(t_l1))
    
    "Filter data based on time"
    time_filter_start = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    time_filter_end = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    mask_vdf = np.logical_and(t_vdf <= time_filter_end, t_vdf>= time_filter_start)
    
    t_mom = np.array(t_mom[mask_vdf])
    n = np.array(n[mask_vdf])
    T = np.array(T[mask_vdf])
    V_srf = np.array(V_srf[mask_vdf])
    V_rtn = np.array(V_rtn[mask_vdf])
    P_SRF = np.array(P_SRF[mask_vdf])
    P_RTN = np.array(P_RTN[mask_vdf])
    # TxTyTz = np.array(TxTyTz[mask_vdf])
    V_solo = np.array(V_solo[mask_vdf])
    rot_matrix = np.array(rot_matrix[mask_vdf])
    
    t_vdf = np.array(t_vdf[mask_vdf])
    vdf = np.array(vdf[mask_vdf])
    qf = np.array(qf[mask_vdf])
    
    P_tensor = np.zeros([len(P_SRF), 3, 3])
    P_tensor_rtn = np.zeros([len(P_RTN), 3, 3])

    for i in range(len(P_SRF)):
        
        P_tensor[i, 0, 0] = P_SRF[i, 0]
        P_tensor[i, 1, 1] = P_SRF[i, 1]
        P_tensor[i, 2, 2] = P_SRF[i, 2]
        P_tensor[i, 0, 1] = P_SRF[i, 3]
        P_tensor[i, 1, 0] = P_SRF[i, 3]
        P_tensor[i, 1, 2] = P_SRF[i, 4]
        P_tensor[i, 2, 1] = P_SRF[i, 4]
        P_tensor[i, 0, 2] = P_SRF[i, 5]
        P_tensor[i, 2, 0] = P_SRF[i, 5]
        
        P_tensor_rtn[i, 0, 0] = P_RTN[i, 0]
        P_tensor_rtn[i, 1, 1] = P_RTN[i, 1]
        P_tensor_rtn[i, 2, 2] = P_RTN[i, 2]
        P_tensor_rtn[i, 0, 1] = P_RTN[i, 3]
        P_tensor_rtn[i, 1, 0] = P_RTN[i, 3]
        P_tensor_rtn[i, 1, 2] = P_RTN[i, 4]
        P_tensor_rtn[i, 2, 1] = P_RTN[i, 4]
        P_tensor_rtn[i, 0, 2] = P_RTN[i, 5]
        P_tensor_rtn[i, 2, 0] = P_RTN[i, 5]
    
    mask_l1 = np.logical_and(t_l1 <= time_filter_end, t_l1>= time_filter_start)
    
    t_l1 = np.array(t_l1[mask_l1])
    counts = np.array(counts[mask_l1])
    
    coords = [phi, theta, energy]
    
    if rtn_rot == False:
        if frame == 'SRF':
            moms = [t_mom, n, T, V_srf, P_tensor]
        elif frame == 'RTN':
            moms = [t_mom, n, T, V_rtn, P_tensor_rtn]
        elif frame == 'Both':
            moms = [t_mom, n, T, V_srf, P_tensor, V_rtn, P_tensor_rtn]
    
    elif rtn_rot == True:
        if frame == 'SRF':
            moms = [t_mom, n, T, V_srf, P_tensor, V_solo, rot_matrix]
        elif frame == 'RTN':
            moms = [t_mom, n, T, V_rtn, P_tensor_rtn, V_solo, rot_matrix]
        elif frame == 'Both':
            moms = [t_mom, n, T, V_srf, P_tensor, V_rtn, P_tensor_rtn, V_solo, rot_matrix]

    vdfs = [t_vdf, vdf, qf]
    l1 = [t_l1, counts]
    
    return coords, moms, vdfs, l1

# fname_vdf = 'Data/2022_02_28/solo_L2_swa-pas-vdf_20220228_V02.cdf'
# fname_mom = 'Data/2022_02_28/solo_L2_swa-pas-grnd-mom_20220228_V02.cdf'
# fname_l1 = 'Data/2022_02_28/solo_L1_swa-pas-3d_20220228_V01.cdf'
# fnames = [fname_vdf, fname_mom, fname_l1]

# date1 = 2022, 2, 28, 13, 30, 0 # starting date and time
# date2 = 2022, 2, 28, 16, 0, 0 # ending date and time

# coords, moms, vdfs, l1 = load_data_PAS_files(fnames, date1, date2, frame = 'Both')

# phi, theta, energy = coords
# t_mom, n, T, V_srf, P_tensor, V_rtn, P_tensor_rtn = moms
# t_vdf, vdf, qf = vdfs
# t_l1, counts = l1

def load_data_MAG_files(fname, date1, date2, frame = 'SRF'):
    "fname contains the name of the file for the VDF, moments and level 1 in that order"

    magnetic_field = cdflib.CDF(fname)
    
    t_B = cdflib.cdfepoch.to_datetime(magnetic_field['Epoch'])
    if frame == 'SRF':
        B = magnetic_field['B_SRF'] # magnetic field in T in spacecraft reference frame.
    if frame == 'RTN':
        B = magnetic_field['B_RTN'] # magnetic field in T in spacecraft reference frame.
    
    t_B = to_datetime(np.array(t_B))
    
    "Filter data based on time"
    time_filter_start = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    time_filter_end = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    mask_B = np.logical_and(t_B <= time_filter_end, t_B >= time_filter_start)
    
    t_B = np.array(t_B[mask_B])
    B = np.array(B[mask_B])
    
    return t_B, B

# fname_Bsrf = 'Data/2022_02_28/solo_L2_mag-srf-normal_20220228_V02.cdf'
# fname_Brtn = 'Data/2022_02_28/solo_L2_mag-rtn-normal_20220228_V02.cdf'
# date1 = 2022, 2, 28, 13, 30, 0 # starting date and time
# date2 = 2022, 2, 28, 16, 0, 0 # ending date and time

# t_B1, B_srf = load_data_MAG_files(fname_Bsrf, date1, date2, frame = 'SRF')
# t_B2, B_rtn = load_data_MAG_files(fname_Brtn, date1, date2, frame = 'RTN')


def load_data_RPW_files(fname, date1, date2):
    "fname contains the name of the file for the VDF, moments and level 1 in that order"

    bia = cdflib.CDF(fname)
    
    t_rpw = cdflib.cdfepoch.to_datetime(bia['Epoch'])
    n_rpw = bia['DENSITY'] # magnetic field in T in spacecraft reference frame.

    t_rpw = to_datetime(np.array(t_rpw))
    
    "Filter data based on time"
    time_filter_start = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    time_filter_end = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    mask_rpw = np.logical_and(t_rpw <= time_filter_end, t_rpw >= time_filter_start)
    
    t_rpw = np.array(t_rpw[mask_rpw])
    n_rpw = np.array(n_rpw[mask_rpw])
    
    return t_rpw, n_rpw

def load_data_PAS_files_multi(fnames_list, date1, date2, frame='SRF', rtn_rot=False):
    """
    Loads and concatenates PAS data across multiple file sets and time range.

    Parameters:
        fnames_list: list of [vdf_file, mom_file, l1_file] sets
        date1: [year, month, day, hour, minute, second] – inclusive start time
        date2: [year, month, day, hour, minute, second] – inclusive end time
        frame: 'SRF', 'RTN', or 'Both'
        rtn_rot: bool

    Returns:
        coords, moms, vdfs, l1 – concatenated and filtered by time
    """
    from collections import defaultdict
    import numpy as np
    import datetime

    # Build datetime boundaries
    t_start = datetime.datetime(*date1)
    t_end = datetime.datetime(*date2)

    all_coords = None
    all_moms = defaultdict(list)
    all_vdfs = defaultdict(list)
    all_l1 = defaultdict(list)

    for fnames in fnames_list:
        try:
            coords, moms, vdfs, l1 = load_data_PAS_files(
                fnames, date1, date2, frame=frame, rtn_rot=rtn_rot
            )
        except Exception as e:
            print(f"⚠️ Error loading files {fnames}: {e}")
            continue

        if all_coords is None:
            all_coords = coords

        for i, val in enumerate(moms):
            all_moms[i].append(val)

        for i, val in enumerate(vdfs):
            all_vdfs[i].append(val)

        for i, val in enumerate(l1):
            all_l1[i].append(val)
            
    # Concatenate
    moms_concat = [np.concatenate(all_moms[i]) for i in sorted(all_moms)]
    vdfs_concat = [np.concatenate(all_vdfs[i]) for i in sorted(all_vdfs)]
    l1_concat = [np.concatenate(all_l1[i]) for i in sorted(all_l1)]

    # Time filtering (after concat)
    def filter_by_time(time_array, *arrays):
        mask = np.logical_and(time_array >= t_start, time_array <= t_end)
        return [time_array[mask]] + [arr[mask] for arr in arrays]

    moms_concat = filter_by_time(moms_concat[0], *moms_concat[1:])
    vdfs_concat = filter_by_time(vdfs_concat[0], *vdfs_concat[1:])
    l1_concat = filter_by_time(l1_concat[0], *l1_concat[1:])

    return all_coords, moms_concat, vdfs_concat, l1_concat


def load_PAS_data_by_daterange(start_date, end_date, frame='SRF', rtn_rot=False, base_path='Data'):
    """
    Wrapper to load PAS data across a date range.

    Parameters:
        start_date: list [YYYY, MM, DD, hh, mm, ss]
        end_date: list [YYYY, MM, DD, hh, mm, ss]
        frame: 'SRF', 'RTN', or 'Both'
        rtn_rot: bool
        base_path: root path to Data folder

    Returns:
        coords, moms, vdfs, l1
    """

    start_dt = datetime.datetime(*start_date[:3])
    end_dt = datetime.datetime(*end_date[:3])
    delta = datetime.timedelta(days=1)

    fnames_list = []

    while start_dt <= end_dt:
        y = start_dt.year
        m = f"{start_dt.month:02d}"
        d = f"{start_dt.day:02d}"
        ymd = f"{y}{m}{d}"
        dir_path = os.path.join(base_path, f"{y}_{m}_{d}")

        try:
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                cdf_files = [f for f in files if f.endswith('.cdf')]
                
                vdf_files = [f for f in cdf_files if 'swa-pas-vdf' in f and ymd in f]
                mom_files = [f for f in cdf_files if 'swa-pas-grnd-mom' in f and ymd in f]
                l1_files = [f for f in cdf_files if 'swa-pas-3d' in f and ymd in f]

                if vdf_files and mom_files and l1_files:
                    fnames_list.append([
                        os.path.join(dir_path, vdf_files[0]),
                        os.path.join(dir_path, mom_files[0]),
                        os.path.join(dir_path, l1_files[0])
                    ])
                    print(f"✓ Found files for {ymd}: {vdf_files[0]}, {mom_files[0]}, {l1_files[0]}")
                else:
                    print(f"⚠️ {ymd}: Missing VDF={len(vdf_files)>0}, MOM={len(mom_files)>0}, L1={len(l1_files)>0}")
                    if cdf_files:
                        print(f"   Available CDF files: {cdf_files}")
            else:
                print(f"⚠️ Directory not found: {dir_path}")
        except Exception as e:
            print(f"⚠️ Error for {ymd}: {e}")

        start_dt += delta

    if not fnames_list:
        print(f"\n❌ No valid file sets found!")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Base path: {base_path}")
        raise FileNotFoundError("❌ No valid file sets found in the given date range.")

    return load_data_PAS_files_multi(fnames_list, start_date, end_date, frame=frame, rtn_rot=rtn_rot)

def load_data_MAG_files_multi(fnames, date1, date2, frame='SRF'):
    """Load and concatenate magnetic field data from multiple files across multiple days."""

    t_B_all = []
    B_all = []

    for fname in fnames:
        try:
            magnetic_field = cdflib.CDF(fname)

            t_B = cdflib.cdfepoch.to_datetime(magnetic_field['Epoch'])
            if frame == 'SRF':
                B = magnetic_field['B_SRF']
            elif frame == 'RTN':
                B = magnetic_field['B_RTN']
            else:
                raise ValueError(f"Unknown frame: {frame}")

            t_B = to_datetime(np.array(t_B))

            time_filter_start = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
            time_filter_end = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])

            mask_B = np.logical_and(t_B >= time_filter_start, t_B <= time_filter_end)

            t_B_all.append(t_B[mask_B])
            B_all.append(B[mask_B])
        except Exception as e:
            print(f"Warning: Could not load file {fname}: {e}")

    if t_B_all:
        t_B_all = np.concatenate(t_B_all)
        B_all = np.concatenate(B_all)
    else:
        t_B_all = np.array([])
        B_all = np.array([])

    return t_B_all, B_all


def load_MAG_data_by_daterange(date1, date2, frame='SRF', base_path='Data'):
    """Wrapper to load magnetic field data over multiple days by finding files with matching patterns."""

    start_date = datetime.datetime(date1[0], date1[1], date1[2])
    end_date = datetime.datetime(date2[0], date2[1], date2[2])

    fnames = []
    current_date = start_date

    while current_date <= end_date:
        y = current_date.year
        m = f"{current_date.month:02d}"
        d = f"{current_date.day:02d}"
        ymd = f"{y}{m}{d}"
        dir_path = os.path.join(base_path, f"{y}_{m}_{d}")

        try:
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                mag_files = [f for f in files if 'mag-srf-normal' in f and ymd in f and f.endswith('.cdf')]

                if mag_files:
                    fnames.append(os.path.join(dir_path, mag_files[0]))
                    print(f"✓ Found MAG file for {ymd}: {mag_files[0]}")
                else:
                    cdf_files = [f for f in files if f.endswith('.cdf')]
                    print(f"⚠️ {ymd}: No MAG file found")
                    if cdf_files:
                        print(f"   Available CDF files: {cdf_files}")
            else:
                print(f"⚠️ Directory not found: {dir_path}")
        except Exception as e:
            print(f"⚠️ Error for {ymd}: {e}")

        current_date += datetime.timedelta(days=1)

    return load_data_MAG_files_multi(fnames, date1, date2, frame=frame)

def load_data_PAS_moments(fnames, date1, date2, frame = 'SRF', rtn_rot = False):
    "fname contains the name of the file for the VDF, moments and level 1 in that order"

    moments = cdflib.CDF(fnames)
    
    n = moments['N'] # plasma density.
    t_mom = cdflib.cdfepoch.to_datetime(moments['Epoch']) #time in epoch reference for moments.
    T = moments['T'] # scalar temperature in eV.
    P_SRF = moments['P_SRF'] # Pressure is symmetric
    P_RTN = moments['P_RTN']
    # TxTyTz = moments['TxTyTz_SRF']
    V_srf = moments['V_SRF'] # velocities in m/s in spacecraft reference frame
    V_rtn = moments['V_RTN']
    
    t_mom = to_datetime(np.array(t_mom))
    # TxTyTz = np.array(TxTyTz)

    # error_vdf = np.array(error_vdf)
    
    "Filter data based on time"
    time_filter_start = datetime.datetime(date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
    time_filter_end = datetime.datetime(date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
    
    mask_vdf = np.logical_and(t_mom <= time_filter_end, t_mom>= time_filter_start)
    
    t_mom = np.array(t_mom[mask_vdf])
    n = np.array(n[mask_vdf])
    T = np.array(T[mask_vdf])
    V_srf = np.array(V_srf[mask_vdf])
    V_rtn = np.array(V_rtn[mask_vdf])
    P_SRF = np.array(P_SRF[mask_vdf])
    P_RTN = np.array(P_RTN[mask_vdf])
    # TxTyTz = np.array(TxTyTz[mask_vdf])
    
    P_tensor = np.zeros([len(P_SRF), 3, 3])
    P_tensor_rtn = np.zeros([len(P_RTN), 3, 3])

    for i in range(len(P_SRF)):
        
        P_tensor[i, 0, 0] = P_SRF[i, 0]
        P_tensor[i, 1, 1] = P_SRF[i, 1]
        P_tensor[i, 2, 2] = P_SRF[i, 2]
        P_tensor[i, 0, 1] = P_SRF[i, 3]
        P_tensor[i, 1, 0] = P_SRF[i, 3]
        P_tensor[i, 1, 2] = P_SRF[i, 4]
        P_tensor[i, 2, 1] = P_SRF[i, 4]
        P_tensor[i, 0, 2] = P_SRF[i, 5]
        P_tensor[i, 2, 0] = P_SRF[i, 5]
        
        P_tensor_rtn[i, 0, 0] = P_RTN[i, 0]
        P_tensor_rtn[i, 1, 1] = P_RTN[i, 1]
        P_tensor_rtn[i, 2, 2] = P_RTN[i, 2]
        P_tensor_rtn[i, 0, 1] = P_RTN[i, 3]
        P_tensor_rtn[i, 1, 0] = P_RTN[i, 3]
        P_tensor_rtn[i, 1, 2] = P_RTN[i, 4]
        P_tensor_rtn[i, 2, 1] = P_RTN[i, 4]
        P_tensor_rtn[i, 0, 2] = P_RTN[i, 5]
        P_tensor_rtn[i, 2, 0] = P_RTN[i, 5]
    
    
    if rtn_rot == False:
        if frame == 'SRF':
            moms = [t_mom, n, T, V_srf, P_tensor]
        elif frame == 'RTN':
            moms = [t_mom, n, T, V_rtn, P_tensor_rtn]
        elif frame == 'Both':
            moms = [t_mom, n, T, V_srf, P_tensor, V_rtn, P_tensor_rtn]
    
    # elif rtn_rot == True:
    #     if frame == 'SRF':
    #         moms = [t_mom, n, T, V_srf, P_tensor, V_solo, rot_matrix]
    #     elif frame == 'RTN':
    #         moms = [t_mom, n, T, V_rtn, P_tensor_rtn, V_solo, rot_matrix]
    #     elif frame == 'Both':
    #         moms = [t_mom, n, T, V_srf, P_tensor, V_rtn, P_tensor_rtn, V_solo, rot_matrix]
    
    return moms

def load_data_PAS_moments_multi(fnames_list, date1, date2, frame='SRF', rtn_rot=False):
    """
    Loads and concatenates PAS data across multiple file sets and time range.

    Parameters:
        fnames_list: list of mom_file sets
        date1: [year, month, day, hour, minute, second] – inclusive start time
        date2: [year, month, day, hour, minute, second] – inclusive end time
        frame: 'SRF', 'RTN', or 'Both'
        rtn_rot: bool

    Returns:
        coords, moms, vdfs, l1 – concatenated and filtered by time
    """
    from collections import defaultdict
    import numpy as np
    import datetime

    # Build datetime boundaries
    t_start = datetime.datetime(*date1)
    t_end = datetime.datetime(*date2)

    all_moms = defaultdict(list)

    for fnames in fnames_list:
        try:
             moms = load_data_PAS_moments(
                fnames, date1, date2, frame=frame, rtn_rot=rtn_rot
            )
        except Exception as e:
            print(f"⚠️ Error loading files {fnames}: {e}")
            continue

        for i, val in enumerate(moms):
            all_moms[i].append(val)
            
    # Concatenate
    moms_concat = [np.concatenate(all_moms[i]) for i in sorted(all_moms)]

    # Time filtering (after concat)
    def filter_by_time(time_array, *arrays):
        mask = np.logical_and(time_array >= t_start, time_array <= t_end)
        return [time_array[mask]] + [arr[mask] for arr in arrays]

    moms_concat = filter_by_time(moms_concat[0], *moms_concat[1:])

    return moms_concat


def load_PAS_moments_by_daterange(start_date, end_date, frame='SRF', rtn_rot=False, base_path='Data'):
    """
    Wrapper to load PAS data across a date range.

    Parameters:
        start_date: list [YYYY, MM, DD, hh, mm, ss]
        end_date: list [YYYY, MM, DD, hh, mm, ss]
        frame: 'SRF', 'RTN', or 'Both'
        rtn_rot: bool
        base_path: root path to Data folder

    Returns:
        coords, moms, vdfs, l1
    """

    start_dt = datetime.datetime(*start_date[:3])
    end_dt = datetime.datetime(*end_date[:3])
    delta = datetime.timedelta(days=1)

    fnames_list = []

    while start_dt <= end_dt:
        y = start_dt.year
        m = f"{start_dt.month:02d}"
        d = f"{start_dt.day:02d}"
        ymd = f"{y}{m}{d}"
        dir_path = os.path.join(base_path, f"{y}_{m}_{d}")

        try:
            # Assume filenames follow consistent structure
            mom_file = os.path.join(dir_path, f"solo_L2_swa-pas-grnd-mom_{ymd}.cdf")

            if os.path.exists(mom_file):
                fnames_list.append(mom_file)
            else:
                print(f"⚠️ Skipping {ymd} — One or more files missing.")
        except Exception as e:
            print(f"⚠️ Error preparing filenames for {ymd}: {e}")

        start_dt += delta

    if not fnames_list:
        raise FileNotFoundError("❌ No valid file sets found in the given date range.")

    return load_data_PAS_moments_multi(fnames_list, start_date, end_date, frame=frame, rtn_rot=rtn_rot)
