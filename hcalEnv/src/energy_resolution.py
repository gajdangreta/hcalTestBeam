# Filename: energy_resolution.py
# Author: Gréta Gajdán 
# Created: 2025-01-21

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys
sys.path.append("../src")

from data_prep import *
from fitting_functions import *

def make_result_df(end_0_params, end_1_params,end_0_errs, end_1_errs, run_numbers, calibration_folder, exclude_high_energy=False):
    run_info=pd.read_csv(calibration_folder+"/runs.csv", index_col="run_number").drop("Unnamed: 0", axis=1)

    params=pd.DataFrame(np.append(np.array(end_0_params),np.array(end_1_params), axis=1),
                        columns=["mu_end0", "sigma_end0", "mu_end1", "sigma_end1"])
    params["run_number"]=np.array(run_numbers)
    params["mu_end0_err"]=np.array([row[0] for row in end_0_errs])
    params["sigma_end0_err"]=np.array([row[1] for row in end_0_errs])
    
    params.set_index(["run_number"], inplace=True)
    
    result_df=params.join(run_info, how='inner')
    result_df["mu_end0"]/=1000
    result_df["mu_end1"]/=1000
    result_df["sigma_end0"]/=1000
    result_df["sigma_end1"]/=1000

    result_df["mu_end0_err"]/=1000
    result_df["sigma_end0_err"]/=1000
    if exclude_high_energy==True:
        low_result_df=result_df[result_df["beam_energy"]<=2]
        return result_df, low_result_df
    else:
        return result_df

def linear_fit(result_df,end, exclude_high_energy=False, low_result_df=None):
    mu="mu_end"+str(int(end))
    
    param, pcov = curve_fit(linear, result_df["beam_energy"], result_df[mu])
    perr=np.sqrt(np.diag(pcov))

    if exclude_high_energy==True:
        param_low, pcov_low = curve_fit(linear, low_result_df["beam_energy"], low_result_df[mu])
        perr_low=np.sqrt(np.diag(pcov_low))
        return param,perr, param_low, perr_low
    else:
        return param, perr

