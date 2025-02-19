# Filename: plot_maker.py
# Author: Gréta Gajdán 
# Created: 2025-02-19

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys
sys.path.append("../src")

from data_prep import *
from fitting_functions import *
from energy_resolution import *

def plot_pulse(pulse_index, run_df, plot_folder, save_plot=False):
    row=run_df.loc[pulse_index]
    end0 = []
    end1 = []
    for j in range(8):
        adc_end0_col = f'adc_{j}_end0'
        adc_end1_col = f'adc_{j}_end1'
        
        end0_val = row[adc_end0_col]
        end1_val = row[adc_end1_col]
        
        end0.append(end0_val)
        end1.append(end1_val)

    plt.scatter(np.arange(0,200,25), end0, color="magenta", s=100, label="end0", ec="magenta", alpha=0.5)
    plt.scatter(np.arange(0,200,25), end1, color="blue", s=100, label="end1", ec="blue", alpha=0.5)
    plt.ylabel("ADC Unit")
    plt.xlabel("Time [ns]")
    plt.title("Event: "+str(int(row["pf_event"]))+" Layer: "+str(int(row["layer"]))+" Bar: "+str(int(row["strip"])))
    plt.legend()
    if save_plot:
        Path(plot_folder+"pulses").mkdir(parents=True, exist_ok=True)
        plt.savefig(p_folder+"pulses/event_"+str(int(row["pf_event"]))+"_layer_"+str(int(row["layer"]))+"_bar_"+str(int(row["strip"]))+".png",
                    bbox_inches="tight")
    plt.show()

def make_energy_error_histograms(run_df, is_relative):
    hand_bins=np.linspace(50,7250,51) #should this be hardcoded?
    wavy, _    = np.histogram(run_df[run_df["problem_type"]=="WAVE"]["adc_sum_end0"], bins=hand_bins)
    mistimed,_ = np.histogram(run_df[run_df["problem_type"]=="LATE"]["adc_sum_end0"], bins=hand_bins)
    spiky,_    = np.histogram(run_df[(run_df["problem_type"]=="SPIKE_0") | 
                              (run_df["problem_type"]=="SPIKE_1")]["adc_sum_end0"], bins=hand_bins)
    if is_relative:
        all_pulses,_=np.histogram(run_df["adc_sum_end0"], bins=hand_bins)
        wavy=wavy/all_pulses
        mistimed=mistimed/all_pulses
        spiky=spiky/all_pulses
    return wavy, mistimed, spiky, hand_bins

def make_layer_error_histogram(run_df, is_relative):
    wavy    =run_df[run_df["problem_type"]=="WAVE"].groupby(["layer"]).count()[["event"]]
    mistimed=run_df[run_df["problem_type"]=="LATE"].groupby(["layer"]).count()[["event"]]
    spiky   =run_df[(run_df["problem_type"]=="SPIKE_0") | (run_df["problem_type"]=="SPIKE_1")].groupby(["layer"]).count()[["event"]]
    if is_relative:
        event_n=run_df.groupby(["layer"]).count()[["event"]]
        wavy=wavy/event_n
        mistimed=mistimed/event_n
        spiky=spiky/event_n
    return wavy, mistimed, spiky

def make_bar_error_histogram(run_df, is_relative):
    wavy=run_df[run_df["problem_type"]=="WAVE"].groupby(["layer","bar"]).count()[["event"]]
    mistimed=run_df[run_df["problem_type"]=="LATE"].groupby(["layer","bar"]).count()[["event"]]
    spiky=run_df[(run_df["problem_type"]=="SPIKE_0") | (run_df["problem_type"]=="SPIKE_1")].groupby(["layer","bar"]).count()[["event"]]
    if is_relative:
        event_n=run_df.groupby(["layer", "bar"]).count()[["event"]]
        wavy=wavy/event_n
        mistimed=mistimed/event_n
        spiky=spiky/event_n
    return wavy, mistimed, spiky

def plot_error_by_energy(run_df, is_relative,plot_folder, save_plot=False):
    # separate the different errors and make histograms
    wavy, mistimed, spiky, hand_bins=make_energy_error_histograms(run_df, is_relative)

    ymax=np.maximum.reduce([wavy, mistimed, spiky], axis=(1,0))*1.05
    plt.rcParams["figure.figsize"]=(12,3)

    plt.subplot(1,3,1)
    plt.hist(hand_bins[:-1],hand_bins,weights= wavy, alpha=0.5, color="dodgerblue", label="Wavy")
    plt.xlabel("Sum ADC", fontsize=12)
    if is_relative:
        plt.ylabel("Chance of error", fontsize=12)
    else:
        plt.ylabel("Number of errors", fontsize=12)
    plt.ylim(0,ymax)
    plt.xlim(0,hand_bins[-1])
    plt.legend(fontsize=12)
    
    plt.subplot(1,3,2)
    plt.hist(hand_bins[:-1],hand_bins,weights= mistimed, alpha=0.5, color="firebrick", label="Mistimed")
    plt.xlabel("Sum ADC", fontsize=12)
    plt.ylim(0,ymax)
    plt.xlim(0,hand_bins[-1])
    plt.legend(fontsize=12)
    
    plt.subplot(1,3,3)
    plt.hist(hand_bins[:-1],hand_bins,weights= spiky, alpha=0.5, color="forestgreen", label="Spiky")
    plt.xlabel("Sum ADC", fontsize=12)
    plt.ylim(0,ymax)
    plt.xlim(0,hand_bins[-1])
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    if save_plot:
        if is_relative:
            plt.savefig(plot_folder+"relative_error_energy_spectrum.png", bbox_inches='tight')
        else:
            plt.savefig(plot_folder+"absolute_error_energy_spectrum.png", bbox_inches='tight')
    plt.show()

def plot_error_by_layer(run_df, is_relative,plot_folder, save_plot=False):
    wavy,mistimed,spiky=make_layer_error_histogram(run_df, is_relative)

    ymax=np.maximum.reduce([wavy, mistimed, spiky], axis=(1,0))*1.05
    plt.rcParams["figure.figsize"]=(12,3)
    plt.subplot(1,3,1)
    plt.bar(wavy.index,wavy["event"],color="dodgerblue",alpha=0.5, label="Wavy")
    plt.xlim(0.5,19.5)
    plt.ylim(0,ymax)
    plt.xticks(np.arange(1,20))
    plt.xlabel("Layer")
    if is_relative:
        plt.ylabel("Chance of error")
    else:
        plt.ylabel("Number of errors")
    plt.legend(loc="upper right", prop={'size': 12})
    
    plt.subplot(1,3,2)
    plt.bar(mistimed.index,mistimed["event"],color="firebrick",alpha=0.5, label="Mistimed")
    plt.xlim(0.5,19.5)
    plt.ylim(0,ymax)
    plt.xticks(np.arange(1,20))
    plt.xlabel("Layer")
    plt.legend(loc="upper right", prop={'size': 12})
    
    plt.subplot(1,3,3)
    plt.bar(spiky.index,spiky["event"],color="forestgreen",alpha=0.5, label="Spiky")
    plt.xlim(0.5,19.5)
    plt.ylim(0,ymax)
    plt.xticks(np.arange(1,20))
    plt.xlabel("Layer")
    plt.legend(loc="upper right", prop={'size': 12})
    
    plt.tight_layout()
    if save_plot:
        if is_relative:
            plt.savefig(plot_folder+"relative_layer_error_spectrum.png", bbox_inches="tight")
        else:
            plt.savefig(plot_folder+"absolute_layer_error_spectrum.png", bbox_inches="tight")
    plt.show()

def plot_error_by_bar(run_df, is_relative, plot_folder,save_plot=False):
    wavy, mistimed, spiky=make_bar_error_histogram(run_df, is_relative)
    ymax=max([max(wavy["event"]), max(mistimed["event"]), max(spiky["event"])])*1.05

    for l in range(1,20):
        plt.suptitle("Layer: "+str(l), fontsize=20)
        
        helper_df=wavy.query('layer == @l').droplevel(0)
        plt.subplot(1,3,1)
        plt.bar(helper_df.index,helper_df["event"],color="dodgerblue",alpha=0.5, label="Wavy")
        plt.xticks(helper_df.index)
        plt.xlabel("Bar")
        plt.ylabel("Chance of error")
        plt.legend(fontsize=12, loc="upper left")
        plt.ylim(0,ymax)
        
        helper_df=mistimed.query('layer == @l').droplevel(0)
        plt.subplot(1,3,2)
        plt.bar(helper_df.index,helper_df["event"],color="firebrick",alpha=0.5, label="Mistimed")
        plt.xticks(helper_df.index)
        plt.xlabel("Bar")
        plt.legend(fontsize=12, loc="upper left")
        plt.ylim(0,ymax)
    
        helper_df=spiky.query('layer == @l').droplevel(0)
        plt.subplot(1,3,3)
        plt.bar(helper_df.index,helper_df["event"],color="forestgreen",alpha=0.5, label="Spiky")
        plt.xticks(helper_df.index)
        plt.xlabel("Bar")
        plt.legend(fontsize=12, loc="upper left")
        plt.ylim(0,ymax)

        if save_plot:
            Path(plot_folder+"error_by_bar").mkdir(parents=True, exist_ok=True)
            if is_relative:
                plt.savefig(plot_folder+"error_by_bar/relative_layer_"+str(l)+".png", bbox_inches='tight')
            else:
                plt.savefig(plot_folder+"error_by_bar/absolute_layer_"+str(l)+".png", bbox_inches='tight')
        plt.show()