# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:11:20 2020

@author: Vivian Imbriotis
"""


import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress




def ramp(X):
    return X*(X>0)


class UnderlineRegressor:
    default_k = 5
    @classmethod
    def loss(cls,params,x, y, k=None, v=False):
        if k is None: k = cls.default_k
        guess = params[0] + params[1] * x
        residuals = y - guess #data points above line are positive
        upper_cost = np.sum(ramp(residuals)**2)
        lower_cost = np.sum(ramp(-residuals)**(2*k))
        cost = upper_cost + lower_cost
        return cost
    @classmethod
    def grad(cls,params,x, y,k=None,v=False):
        if k is None: k = cls.default_k
        guess = params[0] + params[1] * x
        res = y - guess #data points above line are positive
        outer_derivative = (res>0)*(2*res)+(res<0)*(2*k*res**(2*k-1))
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
        with np.errstate(over='raise'):
            while True:
                try:
                    reg = minimize(cls.loss,
                               x0 = start_params,
                               jac = cls.grad,
                               args = (x, y, k),
                               bounds = ((None,None), (0, None)),
                               method = "L-BFGS-B")
                    k+=1
                except FloatingPointError:
                    break
        intercept = reg.x[0] - ramp(np.max(reg.x[0]+x*reg.x[1] - y))
        return (intercept, reg.x[1])
    @classmethod
    def regress_all(cls,X,Y):
        res = []
        for x,y in zip(X,Y):
            res.append(cls.regress(x,y))
        return np.array(res)