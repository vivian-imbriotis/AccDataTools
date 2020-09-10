# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:06:01 2020

@author: viviani
"""
import numpy as np
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d
from scipy.stats import linregress
from scipy.optimize import minimize



# def subtract_neuropil_trace(F, Fneu, alpha = 0.8):
#     subtracted_trace =  F - alpha*Fneu
#     #add values until the lowest point has unit fluorescence
#     return subtracted_trace

def subtract_neuropil_trace(F, Fneu):
    thetas = HubelRegressor.regress_all(Fneu,F)
    betas = thetas[:,1]
    subtracted_trace =  F - Fneu*betas[:,None]
    shifts = ramp(-1*np.min(subtracted_trace, axis = -1))[:,None]
    subtracted_trace += shifts #Shift up explicitylu s.t. no Fcell < 0
    return subtracted_trace

def get_smoothed_running_minimum(timeseries, tau1 = 30, tau2 = 300):
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

def log_transform(dF_on_F):
    m = np.min(dF_on_F)
    return np.log(dF_on_F - (m<0)*(m) + 1e-16)

class HubelRegressor:
    default_k = 100
    @classmethod
    def loss(cls,params, x, y, k=None, delta=None):
        if k is None: k = cls.default_k
        if delta is None:
            delta = np.std(y)
        guess = params[0] + params[1] * x
        residuals = y - guess #data points above line are positive
        upper_cost = (residuals>0)*(delta**2 * (np.sqrt(1 + (residuals/delta)**2)-1))
        lower_cost = (residuals < 0)*(k)*(residuals**(2))
        return np.sum(upper_cost + lower_cost)
    @classmethod
    def grad(cls,params, x, y, k=None, delta = None):
        if k is None: k = cls.default_k
        if delta is None:
            delta = np.std(y)/2
        guess = params[0] + params[1] * x
        residuals = y - guess #data points above line are positive
        per_upper_residual = (residuals > 0)*(residuals / np.sqrt(1+(residuals/delta)**2))
        per_lower_residual = (residuals < 0)*(2*k*residuals)
        outer_derivative  = np.sum(per_upper_residual + per_lower_residual)
        theta_0 = np.sum(outer_derivative)*(-1)
        theta_1 = np.sum(outer_derivative*(-x))
        grad = (theta_0, theta_1)
        return grad
    @classmethod
    def guess(cls,x, y):
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return (intercept, slope)
    @classmethod
    def regress(cls,x, y):
        start_params = cls.guess(x, y)
        k = cls.default_k
        reg = None
        with np.errstate(over='raise', divide = 'raise'):
            while True:
                try:
                    reg = minimize(cls.loss,
                               x0 = start_params,
                               jac = cls.grad,
                               args = (x, y, k),
                               bounds = ((None,None), (0, None)),
                               method = "L-BFGS-B")
                    k+=1
                    break
                except FloatingPointError:
                    if reg: break
                    else: k -= 1
                    
        intercept = reg.x[0] - ramp(np.max(x*reg.x[1] - y))
        return (intercept, reg.x[1])
    @classmethod
    def regress_all(cls,X,Y):
        res = []
        for x,y in zip(X,Y):
            res.append(cls.regress(x,y))
        return np.array(res)
    

def ramp(X):
    return X*(X>0)