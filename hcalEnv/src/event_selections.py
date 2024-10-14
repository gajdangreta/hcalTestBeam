# Filename: event_selections.py
# Author: Gréta Gajdán 
# Created: 2024-10-10

import numpy as np
import pandas as pd

def choose_bar(df,layer, strip):
    df=df[df["strip"]==strip]
    df=df[df["layer"]==layer]
    return df

def drop_columns(df):
    kept_columns=['layer', 'strip', 'pedestal', 'pf_event', 'adc_sum_end0', 'adc_sum_end1', 'end', 'mpv', 'std_dev']
    df=df[df.columns.intersection(kept_columns)]
    return df

def import_data(calibration_folder, data_folder, run_n):
    pedestals=drop_columns(pd.read_csv(calibration_folder+"/pedestals_MIP.csv", sep=','))
    mips=drop_columns(pd.read_csv(calibration_folder+"/mip.csv", sep=','))
    run=drop_columns(pd.read_csv(data_folder+"/run_"+str(run_n)+".csv", sep=','))
    run=run.astype({"adc_sum_end0":float,"adc_sum_end1":float}) 
    return pedestals,mips,run

def confirm_events(df, pedestals, mips):
    # can probably get rid of the loop, but then we would have to split the main df to have ends in different rows. annoying.
    confirmed_data=[]
    layers=np.arange(1,20)
    strips=np.arange(0,12)
    for layer in layers:
        for strip in strips:
            df_slice=choose_bar(df,layer,strip)
            pedestal_slice=choose_bar(pedestals,layer,strip)
            mip=choose_bar(mips,layer,strip)
            if not pedestal_slice.empty:
                df_slice=df_slice[df_slice["adc_sum_end0"]>1.2*pedestal_slice.iloc[0,-2]] # there could be a switch whether we want to
                df_slice=df_slice[df_slice["adc_sum_end1"]>1.2*pedestal_slice.iloc[1,-2]] # require both ends to register the hit

                df_slice.loc[:,"adc_sum_end0"]-=pedestal_slice.iloc[0,-2] # subtracting pedestals
                df_slice.loc[:,"adc_sum_end1"]-=pedestal_slice.iloc[1,-2]

                df_slice.loc[:,"adc_sum_end0"]*=(4.66/mip.iloc[0,-1]) # converting to energy
                df_slice.loc[:,"adc_sum_end1"]*=(4.66/mip.iloc[1,-1])
                
            confirmed_data.extend(df_slice.values.tolist())
    confirmed_df=pd.DataFrame(confirmed_data, columns=['event', 'adc_sum_end0', 'layer', 'strip', 'adc_sum_end1'])
    events_left=confirmed_df.event.unique()
    print("Initial pedestal-based selection performed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return confirmed_df

def confirm_events_sigma(df, pedestals, mips):
    # signal defined as anything above pedestal+5*pedestal std_dev
    confirmed_data=[]
    layers=np.arange(1,20)
    strips=np.arange(0,12)
    for layer in layers:
        for strip in strips:
            df_slice=choose_bar(df,layer,strip)
            pedestal_slice=choose_bar(pedestals,layer,strip)
            mip=choose_bar(mips,layer,strip)
            if not pedestal_slice.empty:
                df_slice=df_slice[df_slice["adc_sum_end0"]>(pedestal_slice.iloc[0,-2]+5*pedestal_slice.iloc[0,-1])] 
                df_slice=df_slice[df_slice["adc_sum_end1"]>(pedestal_slice.iloc[1,-2]+5*pedestal_slice.iloc[0,-1])] 

                df_slice.loc[:,"adc_sum_end0"]-=pedestal_slice.iloc[0,-2] # subtracting pedestals
                df_slice.loc[:,"adc_sum_end1"]-=pedestal_slice.iloc[1,-2]

                df_slice.loc[:,"adc_sum_end0"]*=(4.66/mip.iloc[0,-1]) # converting to energy
                df_slice.loc[:,"adc_sum_end1"]*=(4.66/mip.iloc[1,-1])
                
            confirmed_data.extend(df_slice.values.tolist())
    confirmed_df=pd.DataFrame(confirmed_data, columns=['event', 'adc_sum_end0', 'layer', 'strip', 'adc_sum_end1'])
    events_left=confirmed_df.event.unique()
    print("Initial pedestal-based selection performed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return confirmed_df

def first_layer_cut(df):
    helper_df=df[df['layer']==1].groupby(['event']).sum()
    events_to_remove=helper_df[helper_df['layer']>1].index.values.tolist()
    df=df[~df['event'].isin(events_to_remove)]
    
    events_left=df.event.unique()
    print("Events with multiple hits in first layer removed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return df

def back_layer_cut(df, b_num):
    events_to_remove=df[df['layer']>19-b_num].groupby(['event']).count().index.values.tolist()
    df=df[~df['event'].isin(events_to_remove)]
    
    events_left=df.event.unique()
    print("Events with hits in the back " +str(b_num)+" layers removed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return df

def select_events(df,pedestals, mips, cut_first_layer, cut_back_layers, use_sigma,back_layers):
    print("Number of events: "+str(len(df.pf_event.unique())))
    print("\n")
    if use_sigma==False:
        df=confirm_events(df, pedestals,mips)
    else:
        df=confirm_events_sigma(df, pedestals,mips)
    if cut_first_layer==True:
        df=first_layer_cut(df)
    if cut_back_layers==True:
        df=back_layer_cut(df,back_layers)
    return df

def import_and_select(calibration_folder, data_folder, run_n, cut_first_layer=True, cut_back_layers=True, use_sigma=False,back_layers=7):
    pedestals, mips, run=import_data(calibration_folder, data_folder, run_n)
    run=select_events(run, pedestals, mips,  cut_first_layer, cut_back_layers, use_sigma,back_layers)
    return run