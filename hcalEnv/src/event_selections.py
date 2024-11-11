# Filename: event_selections.py
# Author: Gréta Gajdán 
# Created: 2024-10-10

import numpy as np
import pandas as pd
import scipy.signal as sig

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

#from Axel Helgstrand
def select_faulty_data(data_df, pedestal_df, spike_filter=True, late_trigger_filter=True):
    # loop through all events in the dataframe
    print('filtering faulty events...')
    faulty_data_indecies = []
    for index, row in data_df.iterrows():

        # extract the pulse shape for the event
        end0 = []
        end1 = []

        for j in range(8):

            adc_end0_col = f'adc_{j}_end0'
            adc_end1_col = f'adc_{j}_end1'

            if adc_end0_col not in row or adc_end1_col not in row:
                print(f"Columns '{adc_end0_col}' or '{adc_end1_col}' do not exist in event {row['pf_event']}, skipping")
                continue

            end0_val = row[adc_end0_col]
            end1_val = row[adc_end1_col]

            if np.isnan(end0_val) or np.isnan(end1_val):
                print(f"NaN value found in '{adc_end0_col}' or '{adc_end1_col}' for event {row['pf_event']}, skipping")
                continue

            end0.append(end0_val)
            end1.append(end1_val)

        end0 = np.array(end0)
        end1 = np.array(end1)
        ends = [end0, end1]

        # check the pulse shape
        for end in ends:
            if spike_filter:
                # Check for spikes in data
                # TODO: change threshold to be dependent on the channel pedestal standard deviation.
                peaks, _ = sig.find_peaks((end * -1 + max(end)), prominence=1.5, threshold=10)
                if len(peaks) >= 1:

                    if index not in faulty_data_indecies:
                        faulty_data_indecies.append(index)
                        """plt.plot(end)
                        print('peaks: ', peaks)
                        plt.plot(peaks, np.array(end)[peaks.astype(int)], "x")
                        plt.show()"""

            # check for late triggers in data
            # TODO: this filters away to many good events, make stricter criteria.
            if late_trigger_filter:

                tolerance = 0.2
                if end[-1] >= (end[0] + max(end) * tolerance) or (end[-1] <= end[0] - max(end) * tolerance):

                    tolerance = 0.1
                    if end[1] <= (end[0] + end[0] * tolerance) and end[1] >= (end[0] - end[0] * tolerance):
                        if end[2] <= (end[0] + end[0] * tolerance) and end[2] >= (end[0] - end[0] * tolerance):

                            if index not in faulty_data_indecies:
                                faulty_data_indecies.append(index)

    print('number of faulty events: ', len(faulty_data_indecies))
    return faulty_data_indecies, len(faulty_data_indecies)