# -*- coding: utf-8 -*-

"""
Created on Sun Aug 30 09:59:00 2020

@author: Vivian Imbriotis
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d

from accdatatools.Observations.recordings import Recording

rec = Recording(path)

f = np.stack((rec.F, rec.Fneu),axis=-1)


def underline_cost_func(params, x, y):
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    upper_cost = np.sum(residuals[residuals > 0] / np.sum(residuals > 0)  )
    lower_cost = np.sum(1e6 * (residuals < 0))
    cost = upper_cost + lower_cost
    return cost

def guess(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return [intercept, slope]

def underline_regression(x, y):
    start_params = guess(x, y)

    reg = minimize(underline_cost_func,
               x0 = start_params,
               args = (x, y),
               bounds = ((None, None), (0, None)),
               method = 'Powell' )

    return (reg.x[0], reg.x[1])

def robust_regression(x, y):
    y = y.reshape(-1, 1)
    X = np.vstack((np.ones(y.shape).transpose(), x.reshape(-1, 1).transpose()))
    reg = TheilSenRegressor(random_state=0).fit(X.transpose(), np.ravel(y))

    return (reg.coef_[0], reg.coef_[1])

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



robust_theta = robust_regression(f[:,2], f[:,1])
robust_f = subtract_bg(f[:,1], f[:,2], robust_theta)
robust_df_f = get_df_on_f0(robust_f)

underline_theta = underline_regression(f[:,2], f[:,1])
underline_f = subtract_bg(f[:,1], f[:,2], underline_theta)
underline_df_f = get_df_on_f0(underline_f)


plt.figure()
plt.plot(f[:,2], f[:,1], 'o')
plt.plot(f[:,2], robust_theta[0] + robust_theta[1]*f[:,2], label = 'Robust Regression')
plt.plot(f[:,2], underline_theta[0] + underline_theta[1]*f[:,2], label = 'Underline Regression')
plt.ylabel('Cell Fluorescence')
plt.xlabel('Neuropil Fluorescence')
plt.legend(loc='upper left')
plt.show(block = False)

plt.figure()
plt.subplot(211)
plt.plot(robust_df_f, label = 'Robust Regression')
plt.ylabel('DF/F')
plt.xlabel('Sample')
plt.legend(loc='upper right')

plt.subplot(212)
plt.plot(underline_df_f, label = 'Underline Regression')
plt.ylabel('DF/F')
plt.xlabel('Sample')
plt.legend(loc='upper right')

plt.tight_layout(pad=3.0)
plt.show(block = False)