# Filename: fitting_functions.py
# Author: Gréta Gajdán 
# Created: 2025-01-03

import numpy as np

def gaussian(x, mu, sig, A):
    return (A / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

def linear(x,a,b):
    return a*x+b