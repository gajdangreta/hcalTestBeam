# Filename: make_cuts.py
# Author: Gréta Gajdán 
# Created: 2025-01-03

import numpy as np
import pandas as pd

def isolate_pulse_data(row):
    # Isolates pulse data from one row of dataframe to two numpy arrays. Basically Axel's code.
    # I still find it annoying that the loop is faster than the wildcard lookup.
    end0 = []
    end1 = []

    for j in range(8):
        adc_end0_col = f'adc_{j}_end0'
        adc_end1_col = f'adc_{j}_end1'
        
        end0_val = row[adc_end0_col]
        end1_val = row[adc_end1_col]
        
        end0.append(end0_val)
        end1.append(end1_val)

    end0 = np.array(end0)
    end1 = np.array(end1)
    return end0,end1

def late_filter(end):
    # Axel's late pulse filter, used wholesale. 
    tolerance = 0.2
    if end[-1] >= (end[0] + max(end) * tolerance) or (end[-1] <= end[0] - max(end) * tolerance):

        tolerance = 0.1
        if end[1] <= (end[0] + end[0] * tolerance) and end[1] >= (end[0] - end[0] * tolerance):
            if end[2] <= (end[0] + end[0] * tolerance) and end[2] >= (end[0] - end[0] * tolerance):
                return 1

    return 0

def does_pulse_dip(end, end_std, layer, bar, which):
    # Does the pulse (as measured in ONE end) have an unexpected dip.

    # I think this doesn't catch dips that have a perfectly flat bottom
    end_diff=np.diff(end)
    end_sign=np.sign(end_diff)
    end_sdiff=np.diff(end_sign)

    dip_loc=np.isin(end_sdiff,2)
    dip_loc=np.append(dip_loc,False)
    dip_value=abs(end_diff[dip_loc])

    if dip_value.size==1 and dip_value>5*end_std:
        return 1
    elif dip_value.size>1 and dip_value.any()>5*end_std:
        print("Something went wrong.") # Sometimes we have multiple dips. 
        return 1
        
    return 0

def categorize_pulses(row):
    out_line=[row.name]
    
    # isolating datapoints from row
    end0, end1=isolate_pulse_data(row)

    # preparing to filter
    layer=row["layer"]    # used as variables to avoid frequent lookup
    bar=row["strip"]
    event=row["pf_event"]

    end0_std=row["pedestal_per_time_sample_std_dev_end0"]
    end1_std=row["pedestal_per_time_sample_std_dev_end1"]

    out_line.append(layer)
    out_line.append(bar)
    out_line.append(event)

    # spike and dip filter
    
    dip_end0=does_pulse_dip(end0, end0_std, layer, bar, 0)
    dip_end1=does_pulse_dip(end1, end1_std, layer, bar, 1)

    if dip_end0==1 and dip_end1==1:
        # Wavy pulse
        out_line.append(1)
        out_line.append("WAVE")
    elif dip_end0==1 and dip_end1==0:
        # Spike in end0
        out_line.append(1)
        out_line.append("SPIKE_0")
    elif dip_end0==0 and dip_end1==1:
        # Spike in end1
        out_line.append(1)
        out_line.append("SPIKE_1")
    else:
        # No dipping
        late_end0=late_filter(end0)
        late_end1=late_filter(end1)
    
        if late_end0==1 or late_end1==1:
            # Late pulse
            out_line.append(1)
            out_line.append("LATE")
        else:
            # Good pulse
            out_line.append(0)
            out_line.append("NaN")

    return out_line

def make_pulse_profiles(run_df):
    problems=run_df.apply(categorize_pulses, axis=1).values.tolist()
    
    pulse_df=pd.DataFrame(problems, columns=["index","layer", "bar", "event", "has_problem", "problem_type"])
    pulse_df.set_index("index", inplace=True)
    
    return pulse_df