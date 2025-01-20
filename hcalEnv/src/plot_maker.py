# Filename: plot_maker.py
# Author: Gréta Gajdán 
# Created: 2025-01-07

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys
sys.path.append("../src")

from data_prep import *
from fitting_functions import *

def longitudinal_profile(run_df, run_number, plot_folder, save_plot=False):
    # generate the longitudinal shower profile of a run. 
    # only works on dataframes where energy conversion had already happened
    
    n_event=len(run_df.pf_event.unique())
    layer_sum=run_df.groupby(["layer"]).sum()[["adc_sum_end0", "adc_sum_end1"]]/n_event

    plt.bar(layer_sum.index-0.2, layer_sum["adc_sum_end0"], color="magenta", alpha=0.5, edgecolor="magenta", label="end_0", width=0.4)
    plt.bar(layer_sum.index+0.2, layer_sum["adc_sum_end1"], color="blue", alpha=0.5, edgecolor="blue", label="end_1", width=0.4)
    plt.xlim(0.5,19.5)
    plt.xticks(np.arange(1,20))
    plt.legend()
    plt.xlabel("Layer", fontsize=12.5)
    plt.ylabel("Deposited energy [MeV]", fontsize=12.5)
    plt.title("Run "+str(run_number), fontsize=15)
    
    if save_plot==True:
        plt.savefig(plot_folder+"longitudinal_profile.png", bbox_inches='tight')
        
    plt.show()

def transverse_profile(run_df,plot_folder, save_plot=False):
    # generate the transverse shower profile of a run. 
    # only works on dataframes where energy conversion had already happened
    
    n_event=len(run_df.pf_event.unique())
    strip_sum=run_df.groupby(["layer", "strip"]).sum()[["adc_sum_end0", "adc_sum_end1"]]/n_event
    
    y_max=strip_sum.max(axis=None)
    
    for l in range(1,20):
        if l in run_df['layer'].values:
            helper_df=strip_sum.xs(l, level=0, axis=0, drop_level=True)
            
            plt.bar(helper_df.index-0.2, helper_df["adc_sum_end0"], color="magenta", alpha=0.5, edgecolor="magenta", label="end_0", width=0.4)
            plt.bar(helper_df.index+0.2, helper_df["adc_sum_end1"], color="blue", alpha=0.5, edgecolor="blue", label="end_1", width=0.4)
            plt.legend()
            plt.xticks(helper_df.index)
            plt.ylim(5e-4,y_max+1)
            plt.yscale('log')
            plt.xlabel("Bar", fontsize=12.5)
            plt.ylabel("Deposited energy [MeV]", fontsize=12.5)
            plt.title("Layer "+str(l), fontsize=15)
            if save_plot==True:
                Path(plot_folder+"transverse_profile").mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_folder+"transverse_profile/transverse_layer_"+str(l)+".png", bbox_inches='tight')
            plt.show()

def make_shower_profiles(run_df, mip_df, run_number, plot_folder):
    # generate and save both transverse and longitudinal shower profiles of a run
    run_df=convert_to_MeV(run_df, mip_df, is_it_pulsed=False)
    longitudinal_profile(run_df,run_number,plot_folder, save_plot=True)
    transverse_profile(run_df,plot_folder, True)
    return

def make_gaussian_fit_plot(run_df, mip_df, run_number, plot_folder, save_plot=False):
    run_df=convert_to_MeV(run_df,mip_df, False)
    grouped_df=run_df.groupby(['pf_event']).sum()[['adc_sum_end0', 'adc_sum_end1']]

    h0, bins=np.histogram(grouped_df["adc_sum_end0"], bins=20)
    h1, _=np.histogram(grouped_df["adc_sum_end1"], bins=bins)

    idx_max = list(h0).index(max(h0))
    mu_p0=bins[idx_max]

    bin_fit=(bins[:-1] + bins[1:]) / 2
    
    param0, _=curve_fit(gaussian,bin_fit, h0, p0=[mu_p0, 100,6000])
    param1, _=curve_fit(gaussian,bin_fit, h1, p0=[mu_p0, 100,6000])

    #plt.bar(bins[:-1]-1.4, h0,width=2.4, align="edge", color="magenta", edgecolor="magenta", alpha=0.5, label="end_0")
    #plt.bar(bins[:-1]+1.4, h1,width=2.4, align="edge", color="blue", edgecolor="blue", alpha=0.5, label="end_1")
    plt.hist(grouped_df, bins=20,color=["magenta", "blue"], edgecolor="black", alpha=0.5, label=["end_0", "end_1"])
    
    plt.plot(np.linspace(bins[0], bins[-2], 1000), gaussian(np.linspace(bins[0], bins[-2], 1000), *param0),
             color="darkmagenta", linestyle="dashed",
             label=r"end_0 gaussian fit:""\n"" $\mu$="+str(round(param0[0],2))+", $\sigma$="+str(round(param0[1],2)))
    plt.plot(np.linspace(bins[0], bins[-2], 1000), gaussian(np.linspace(bins[0], bins[-2], 1000), *param1),
             color="darkblue", linestyle="dotted", 
             label=r"end_1 gaussian fit:""\n"" $\mu$="+str(round(param1[0],2))+", $\sigma$="+str(round(param1[1],2)))
    plt.title("Run "+str(run_number), fontsize=15)
    plt.xlabel("Reconstructed energy [MeV]")
    plt.ylabel("Number of events")
    plt.legend()
    
    if save_plot==True:
        plt.savefig(plot_folder+"reconstructed_energy.png", bbox_inches='tight')
    
    plt.show()
    return param0, param1