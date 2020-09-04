# -*- coding: utf-8 -*-

"""
Created on Sun Aug 30 09:59:00 2020

@author: Vivian Imbriotis
"""

import numpy as np
import seaborn as sns
from scipy.optimize import minimize, check_grad, approx_fprime
from scipy.stats import linregress, norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import TheilSenRegressor
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d

from accdatatools.Observations.recordings import Recording
import pickle as pkl

#Load a recording
try:
    with open(r"C:/Users/viviani/desktop/cache.pkl","rb") as file:
        rec = pkl.load(file)
except (FileNotFoundError, EOFError):
    with open(r"C:/Users/viviani/desktop/cache.pkl","wb") as file:
        rec = Recording(r"D:\Local_Repository\CFEB013\2016-05-31_02_CFEB013")
        pkl.dump(rec,file)

def heaviside(X,k=200):
    '''
    Analytic approximation of the heaviside function.
    Approaches Heaviside(x) as K goes to +inf
    '''
    return 1/(1+np.exp(-2*k*X))

def ramp(X):
    return X*(X>0)

def squash(X,a=100):
    '''
    Squashes X from positive reals to the [0,1) half-open
    interval.
    '''
    return 1 - np.exp(-(X/a)**2)


def asymmetric_ramp_loss(params, x, y):
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    upper_cost = np.sum(residuals[residuals > 0] / np.sum(residuals > 0)  )
    lower_cost = np.sum((residuals < 0))
    cost = upper_cost + lower_cost
    return cost

def asymmetric_ramp_grad(params, x, y, diracs = False):
    N = len(x) if hasattr(x,"__len__") else 1
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    if diracs:
        theta_0 = -np.sum((residuals > 0) + K*dirac(residuals))
        theta_1 = -np.sum(x*[residuals > 0] + x*K*dirac(residuals))
    else:
        theta_0 = -np.sum((residuals > 0))
        theta_1 = -np.sum(x*[residuals > 0])
    return (theta_0, theta_1)

K = 1
    
def squashed_loss(params, x = rec.Fneu[4], y = rec.F[4], v=True, k = 1, a = 1):
    '''
    As a one-liner this is
    (1-exp(-ramp(x)**2)) + N*1/(1 + exp(2Kx)) + ramp(-x)
    where x is a residual, summed for all residuals
    '''
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    residuals_cost = np.sum(squash(ramp(residuals),a))
    negativity_cost = np.sum(x.shape*heaviside(-residuals,k) + ramp(-residuals))
    cost = residuals_cost + negativity_cost
    return cost

def squashed_grad(params, x = rec.Fneu[4], y = rec.F[4], k = 1, a = 1):
    N = len(x) if hasattr(x,"__len__") else 1
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    term1 = 2*residuals*np.exp(-((np.sum(ramp(residuals)))/a)**2)/(a**2)*heaviside(residuals,k)
    log_term2 = np.log(2*k*N) + 2*k*residuals - 2*np.log(1+np.exp(2*k*residuals))
    term2 = np.exp(log_term2)
    term3 = heaviside(residuals,k)
    outer_derivative = term1 + term2 + term3
    theta_0 = np.sum(outer_derivative) * (-1)
    theta_1 = np.sum(outer_derivative * (-x))
    return (theta_0, theta_1)

def quadratic_loss(params,x = rec.Fneu[4], y = rec.F[4],k=4,v=False):
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    upper_cost = np.sum(ramp(residuals)**2)
    lower_cost = np.sum(ramp(-residuals)**(2*k))
    cost = upper_cost + lower_cost
    return cost

def quadratic_grad(params,x = rec.Fneu[4], y = rec.F[4],k=4,v=False):
    guess = params[0] + params[1] * x
    res = y - guess #data points above line are positive
    outer_derivative = (res>0)*(2*res)+(res<0)*(2*k*res**(2*k-1))
    theta_0 = np.sum(outer_derivative)*(-1)
    theta_1 = np.sum(outer_derivative*(-x))
    grad = (theta_0, theta_1)
    return grad

def normalize(x,y):
    vector = np.array((x,y))
    norms = np.linalg.norm(vector,axis=0)
    return vector / norms


dirac = norm(0,1e-2).pdf


def guess(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return (intercept, slope)

class ParabolicRegressor:
    default_k = 30
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
        reg = None
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
                    if reg: break
                    else: k -= 1
                    
        intercept = reg.x[0] - ramp(np.max(reg.x[0]+x*reg.x[1] - y))
        return (intercept, reg.x[1])


class HubelRegressor:
    default_k = 30
    @classmethod
    def loss(cls,params, x, y, k=None, delta=None):
        if k is None: k = cls.default_k
        if delta is None:
            delta = np.std(y)/2
        guess = params[0] + params[1] * x
        residuals = y - guess #data points above line are positive
        upper_cost = (residuals>0)*(delta**2 * (np.sqrt(1 + (residuals/delta)**2)-1))
        lower_cost = (residuals < 0)*(residuals**(2*k))
        return np.sum(upper_cost + lower_cost)
    @classmethod
    def grad(cls,params, x, y, k=None, delta = None):
        if k is None: k = cls.default_k
        if delta is None:
            delta = np.std(y)/2
        guess = params[0] + params[1] * x
        residuals = y - guess #data points above line are positive
        per_upper_residual = (residuals > 0)*(residuals / np.sqrt(1+(residuals/delta)**2))
        per_lower_residual = (residuals < 0)*(2*k*residuals**(2*k-1))
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
                except FloatingPointError:
                    if reg: break
                    else: k -= 1
                    
        intercept = reg.x[0] - ramp(np.max(reg.x[0]+x*reg.x[1] - y))
        return (intercept, reg.x[1])

def underline_regression(x, y, method = "ramp"):
    start_params = guess(x, y)
    if method=="ramp":
        reg = minimize(asymmetric_ramp_loss,
                   x0 = start_params,
                   args = (x, y),
                   bounds = ((None, None), (0, None)),
                   method = "Powell")
    elif method == 'quadratic' or method == "parabolic":
        reg = ParabolicRegressor.regress(x,y)
        return reg
    elif method == "squashed":
        reg = minimize(squashed_loss,
                   x0 = start_params,
                   jac = squashed_grad,
                   args = (x, y),
                   bounds = ((None,None), (0, 1)),
                   method = "L-BFGS-B")
    elif method == "median":
        y = y.reshape(-1, 1)
        X = np.vstack((np.ones(y.shape).transpose(), x.reshape(-1, 1).transpose()))
        reg = TheilSenRegressor(random_state=0).fit(X.transpose(), np.ravel(y))
        offset = np.min(subtract_bg(y, x, [reg.coef_[0],reg.coef_[1]]))
        return np.array([reg.coef_[0] + offset, reg.coef_[1]])
    elif method == "huber":
        reg = HubelRegressor.regress(x,y)
        return reg
    return (reg.x[0], reg.x[1])


def robust_regression(x, y):
    y = y.reshape(-1, 1)
    X = np.vstack((np.ones(y.shape).transpose(), x.reshape(-1, 1).transpose()))
    reg = TheilSenRegressor(random_state=0).fit(X.transpose(), np.ravel(y))

    return (reg.coef_[0], reg.coef_[1])

def subtract_bg(f, bg, theta):
    return f - theta[0] - (theta[1] * bg)

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
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # color[0] = 'blue'
    # color[1] = 'orange'
    # color[2] = 'green'
    # color[3] = 'red'
    def __init__(self, F, Fneu, methods):
        sns.set_style("dark")
        sns.set_context("paper")
        
        thetas = []
        df_traces = []
        for method in methods:
            theta = underline_regression(Fneu, F, method)
            underline_f = subtract_bg(F, Fneu, theta)
            print(np.mean(underline_f))
            underline_df_f = get_df_on_f0(underline_f)
            thetas.append(theta)
            df_traces.append(underline_df_f)
        
        self.fig = plt.figure(figsize=(8,4),tight_layout=True)
        self.axes = []
        regression_axis = self.fig.add_axes((0.075,0.1,0.4,0.85))
        self.axes.append(regression_axis)
        regression_axis.plot(Fneu, F, 'o', color = self.color[0])
        regression_axis.set_ylabel('Cell Fluorescence')
        regression_axis.set_xlabel('Neuropil Fluorescence')
        method_axes_height = 0.9/len(methods)
        for idx, (color, df_trace, theta, method) in enumerate(zip(self.color[1:],df_traces,thetas,methods)):
            print(theta)
            regression_axis.plot(Fneu, theta[0] + theta[1]*Fneu, label = method,
                                 color = color)
            method_axis = self.fig.add_axes((0.55,0.05 + idx*method_axes_height,
                                             0.40,method_axes_height))
            method_axis.plot(df_trace, label = method,
                          color = color)
            method_axis.set_ylabel('Predicted DF/F')
            method_axis.set_xticks([])
            method_axis.legend(loc='upper right')
            self.axes.append(method_axis)
            
    def show(self):
        self.fig.show()

def plot_loss_functions():
    X = np.linspace(-3,3,1000)
    params = (0,0)
    ramp_loss = [asymmetric_ramp_loss(params,np.array(0),np.array(x)) for x in X]
    quad_loss = [quadratic_loss(params,np.array(0),np.array(x),k=10) for x in X]
    hubel_loss = [HubelRegressor.loss(params,np.array(0),np.array(x),k=10, delta=0.5) for x in X]
    squashed = [squashed_loss(params,np.array(0),np.array(x)) for x in X]
    print(hubel_loss)
    fig, ax = plt.subplots(ncols = 4)
    ax[0].plot(X,ramp_loss)
    ax[1].plot(X,quad_loss)
    ax[2].plot(X,hubel_loss)
    ax[3].plot(X,squashed)
    for a in ax:
        a.set_ylim((0,6))
    fig.show()



plt.close('all')
plot_loss_functions()
# UnderlineRegressionFigure(rec.F[57],rec.Fneu[57], methods = ("ramp","quadratic","huber")).show()