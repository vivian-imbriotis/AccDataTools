# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:30:00 2020

@author: Vivian Imbriotis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d
from sklearn.linear_model import TheilSenRegressor

np.random.seed(42)

def make_data(n_points = 10000):
    spikes = make_spikes(n_points)
    bg = make_bg(n_points)
    data = {}
    data["frame_n"] = np.arange(0, n_points)
    data["ground_truth"] = spikes
    data["bg"] = bg
    data["slope"] = np.random.random()+0.5
    data["raw"] = data["ground_truth"] + data["slope"]*bg
    return data

def make_spikes(n_points = 10000):
    spike_prob = 0.01
    spike_size_mean = 1
    kernel = make_kernel()
    spike_times = 1.0 * (np.random.rand(n_points-kernel.size+1) < spike_prob)
    spike_amps = np.random.lognormal(mean = spike_size_mean, sigma = 0.5, size=spike_times.size) * spike_times
    spikes = np.convolve(spike_amps, kernel) + np.random.randn(n_points) / 2 + 10
    return spikes

def make_bg(n_points = 10000):
    return np.sin(np.arange(0, n_points)/(6.28*10))/2 + np.random.randn(n_points)/2


def make_kernel(tau = 10):
    length = tau*8
    time = np.arange(0, length)
    return np.exp(-time/tau)
   

#Just plotting and helper functions below


def plot_data():
    data = make_data()

    data = proc_data(data)
   
    plt.figure(figsize = (15, 9))
    plt.subplot(321)
    plt.plot(data["frame_n"], data["raw"])
    plt.ylabel('Raw F')
    plt.xlabel('Frame Number')
   
    plt.subplot(322)
    plt.plot(data["bg"], data["raw"], 'o')
    plt.plot(data["bg"], data["NRR_theta"][0] + 10*data["NRR_theta"][1] + data["NRR_theta"][1]*data["bg"], label = 'Naive Robust Regression')
    plt.plot(data["bg"], data["RR_theta"][0] + data["RR_theta"][1]*data["bg"], label = 'Robust Regression')

 
    plt.ylabel('Total Fluorescence')
    plt.xlabel('Neuropil Fluorescence')
    plt.legend(loc='upper left')
 
    plt.subplot(323)
    plt.plot(data["frame_n"], data["ground_truth"])
    plt.ylabel("Ground Truth")
   
    plt.subplot(324)
    plt.plot(data["frame_n"], data["NRR_df_on_f"])
    plt.ylabel("Naive Robust Reg df/f")

    plt.subplot(325)
    plt.plot(data["frame_n"], data["RR_df_on_f"])
    plt.ylabel("Robust Regression df/f")

    plt.subplot(326)
    plt.plot(data["ground_truth"], data["RR_df_on_f"], 'o')
    plt.plot(data["ground_truth"], data["ground_truth"], label= 'Unity')
    plt.title("MeanSquareError = " + str(calc_error(data)))
    plt.xlabel("Ground Truth")
    plt.ylabel("Robust Regression Aproach")
    plt.legend(loc='upper left')
   

   

plt.show(block = False)

def proc_data(data):
    data["NRR_theta"] = naive_robust_regression(data["bg"], data["raw"])
    data["NRR_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["NRR_theta"])
    data["NRR_df_on_f"] = get_df_on_f0(data["NRR_bg_subtracted"])

    data["RR_theta"] = robust_regression(data["bg"], data["raw"])
    data["RR_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["RR_theta"])
    data["RR_df_on_f"] = get_df_on_f0(data["RR_bg_subtracted"])
    return data

def calc_error(data):
    return np.sum((data["ground_truth"] - data["RR_df_on_f"])**2)/data["ground_truth"].size

def ols(x,y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return np.array([intercept, slope])

def naive_robust_regression(x, y):
    y = y.reshape(-1, 1)
    X = np.vstack((np.ones(y.shape).transpose(), x.reshape(-1, 1).transpose()))
    reg = TheilSenRegressor(random_state=0).fit(X.transpose(), np.ravel(y))

    return np.array([reg.coef_[0], reg.coef_[1]])

def robust_regression(x, y):
    y_reshape = y.reshape(-1, 1)
    X = np.vstack((np.ones(y_reshape.shape).transpose(), x.reshape(-1, 1).transpose()))
    reg = TheilSenRegressor(random_state=0).fit(X.transpose(), np.ravel(y_reshape))

    subtracted_data = subtract_bg(y, x, [reg.coef_[0], reg.coef_[1]])
    offset = np.min(subtracted_data)

    return np.array([reg.coef_[0]+offset, reg.coef_[1]])



def subtract_bg(f, bg, theta):
    return f - ( theta[0] + theta[1] * bg)

def get_smoothed_running_minimum(timeseries, tau1 = 30, tau2 = 100):
    result = minimum_filter1d(uniform_filter1d(timeseries,tau1,mode='nearest'),
                            tau2,
                            mode = 'reflect')
    return result

def get_df_on_f0(F,F0=None):
    if type(F0)!=type(None):
        return (F - F0) / F0
    else:
        F0 = get_smoothed_running_minimum(F)
        return get_df_on_f0(F,F0)




plot_data()