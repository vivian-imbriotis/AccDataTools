# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:06:01 2020

@author: viviani
"""
import numpy as np
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d
from accdatatools.ProcessFluorescence.vectorised_underline_regression import UnderlineRegressor


# def subtract_neuropil_trace(F, Fneu, alpha = 0.8):
#     subtracted_trace =  F - alpha*Fneu
#     #add values until the lowest point has unit fluorescence
#     return subtracted_trace - np.min(subtracted_trace) + 1

def subtract_neuropil_trace(F, Fneu):
    alpha,beta = UnderlineRegressor.regress_all(Fneu,F)
    subtracted_trace =  F - Fneu*beta[:,None] - alpha[:,None]
    return subtracted_trace

def get_smoothed_running_minimum(timeseries, tau1 = 30, tau2 = 100):
    mode = 'nearest'
    result = minimum_filter1d(uniform_filter1d(timeseries,tau1,mode=mode),
                            tau2,
                            mode = 'reflect')
    return result

def get_df_on_f0(F,F0=None):
    if type(F0)!=type(None):
        return (F - F0) / F0
    else:
        F0 = get_smoothed_running_minimum(F)
        return get_df_on_f0(F,F0)
    
    


def underline_cost_func(params, x, y):
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    upper_cost = np.sum(residuals[residuals > 0]**2 / np.sum(residuals > 0)  )
    lower_cost = np.sum(1e20 * (residuals < 0))
    cost = upper_cost + lower_cost
    return cost


def guess(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return (intercept, slope)

def underline_regression(x, y, method = "Powell"):
    start_params = guess(x, y)

    reg = minimize(underline_cost_func,
               x0 = start_params,
               args = (x, y),
               bounds = ((None, None), (0, None)),
               method = method )

    return (reg.x[0], reg.x[1])


def subtract_neuropil_trace_using_regression(f, bg, theta):
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


class UnderlineRegressionFigure:
    def __init__(self, recording, roi_number, method = 'robust'):
        sns.set_style("dark")
        sns.set_context("paper")
        robust = method in ('robust','seigel','theil sen')
        f = np.stack((np.zeros(recording.F[roi_number,:].shape),
                      recording.F[roi_number,:], 
                      recording.Fneu[roi_number,:]
                              ),axis=-1)
        if robust:
            robust_theta = robust_regression(f[:,2], f[:,1])
            robust_f = subtract_bg(f[:,1], f[:,2], robust_theta)
            robust_df_f = get_df_on_f0(robust_f)
        else:
            alt_theta = underline_regression(f[:,2], f[:,1],method=method)
            alt_f = subtract_bg(f[:,1], f[:,2], alt_theta)
            alt_df_f = get_df_on_f0(alt_f)
            
        
        underline_theta = underline_regression(f[:,2], f[:,1])
        underline_f = subtract_bg(f[:,1], f[:,2], underline_theta)
        underline_df_f = get_df_on_f0(underline_f)
    
        
        
        self.fig = plt.figure(figsize=(8,4),tight_layout=True)
        
        regression_axis = self.fig.add_axes((0.075,0.1,0.4,0.85))
        regression_axis.plot(f[:,2], f[:,1], 'o')
        if robust:
            regression_axis.plot(f[:,2], robust_theta[0] + robust_theta[1]*f[:,2], label = 'Theil-Sen Estimator Regression')
        else:
            regression_axis.plot(f[:,2], alt_theta[0] + alt_theta[1]*f[:,2], label = f'Underline Regression with {method} method')
        regression_axis.plot(f[:,2], underline_theta[0] + underline_theta[1]*f[:,2], label = 'Underline Regression with Powell method')
        regression_axis.set_ylabel('Cell Fluorescence')
        regression_axis.set_xlabel('Neuropil Fluorescence')
        regression_axis.legend(loc='upper left')
        regression_axis.set_yticks(np.fix(np.array(regression_axis.get_ylim())/5)*5)
        
        alt_axis = self.fig.add_axes((0.55,0.5,0.4,0.4))
        if robust:
            alt_axis.plot(robust_df_f, label = 'Theil-Sen Estimator Regression')

        else:
            alt_axis.plot(alt_df_f, label = f'Underline Regression with {method} Method')
        alt_axis.set_ylabel('DF/F')
        # theilsen_axis.set_xlabel('Sample')
        alt_axis.set_xticks([])
        alt_axis.legend(loc='upper right')
            
        
        underline_axis = self.fig.add_axes((0.55,0.1,0.4,0.4))
        underline_axis.plot(underline_df_f, label = 'Underline Regression')
        underline_axis.set_ylabel('DF/F')
        underline_axis.set_xlabel('Sample')
        underline_axis.set_xticks([])
        underline_axis.legend(loc='upper right')
        underline_axis.set_yticks(np.fix(np.array(underline_axis.get_ylim())/5)*5)



    def show(self):
        self.fig.show()
