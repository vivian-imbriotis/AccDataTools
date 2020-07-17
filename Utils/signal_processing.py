# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:10:30 2020

@author: viviani
"""

import numpy as np

def rising_edges(array, cutoff = 2.5):
    '''
    Detect rising edges.

    Parameters
    ----------
    array : arraylike of float
        Signal array of voltages.
    cutoff : int, optional
        The cutoff voltage to distinguish digital True from False. 
        The default is 2.5.

    Returns
    -------
    Arraylike of Bool
        True on rising edge indicies of input array, else false
    '''
    digital_array = np.greater(array,cutoff)
    shifted_array = np.logical_not(np.append(digital_array[0],digital_array[:-1]))
    return np.logical_and(digital_array,shifted_array)

def falling_edges(array, cutoff = 2.5):
    '''
    Detect falling edges.

    Parameters
    ----------
    array : arraylike of float
        Signal array of voltages.
    cutoff : int, optional
        The cutoff voltage to distinguish digital True from False. 
        The default is 2.5.

    Returns
    -------
    Arraylike of Bool
        True on falling edge indicies of input array, else False
    '''
    digital_array = np.greater(array,cutoff)
    shifted_array = np.logical_not(np.append(digital_array[1:],digital_array[-1]))
    return np.logical_and(digital_array,shifted_array)

def rising_or_falling_edges(array, cuttoff=2.5):
    rising  = rising_edges(array,cuttoff)
    falling = falling_edges(array,cuttoff)
    return np.logical_or(rising,falling)