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

DEFAULT_K = 20


#Load a recording
try:
    with open(r"C:/Users/viviani/desktop/cache.pkl","rb") as file:
        rec = pkl.load(file)
except (FileNotFoundError, EOFError):
    with open(r"C:/Users/viviani/desktop/cache.pkl","wb") as file:
        rec = Recording(r"D:\Local_Repository\CFEB013\2016-05-31_02_CFEB013")
        pkl.dump(rec,file)



def ramp(X):
    return X*(X>0)

dirac = norm(0,10).pdf


def original_loss(params, x, y):
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    upper_cost = np.sum(residuals[residuals > 0] / np.sum(residuals > 0)  )
    lower_cost = np.sum(K * (residuals < 0))
    cost = upper_cost + lower_cost
    return cost


def original_grad(params, x, y, diracs = False):
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
    


def qloss(params,x = rec.Fneu[4], y = rec.F[4],k=DEFAULT_K,v=False):
    guess = params[0] + params[1] * x
    residuals = y - guess #data points above line are positive
    upper_cost = np.sum(ramp(residuals)**2)
    lower_cost = np.sum(ramp(-residuals)**(2*k))
    cost = upper_cost + lower_cost
    return cost

def qgrad(params,x = rec.Fneu[4], y = rec.F[4],k=DEFAULT_K,v=False):
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

def plot_quadratic_loss(i):
    plt.close('all')
    X = np.arange(-200, 300, 12.5)
    Y = np.arange(-5, 5, 0.25)
    fig2, ax = plt.subplots()
    X, Y = np.meshgrid(X, Y)
    U,V = np.vectorize(lambda x,y:qgrad(
        (x,y),rec.Fneu[i,0],rec.F[i,0]))(X,Y)
    U,V = normalize(U,V)
    ax.quiver(X,Y,-U,-V, angles = "xy")
    ax.set_xlabel("Intercept")
    ax.set_ylabel("Slope")
    ax.set_title("-Grad(Loss)")
    Z = np.vectorize(lambda x,y:qloss((x,y),rec.Fneu[i,0],rec.F[i,0],v=True))(X,Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,np.log(Z))
    ax.set_xlabel("Intercept")
    ax.set_ylabel("Slope")
    ax.set_zlabel("Log Loss")
    ax.set_title("Loss Function")
    plt.show()



def guess(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return (intercept, slope)

def underline_regression(x, y, method = "Powell"):
    start_params = guess(x, y)
    if method=="Powell":
        reg = minimize(original_loss,
                   x0 = start_params,
                   args = (x, y),
                   bounds = ((None, None), (0, None)),
                   method = "Powell")
    elif method in ("BFGS","L-BFGS-B"):
        reg = minimize(qloss,
                   x0 = start_params,
                   jac = qgrad,
                   args = (x, y),
                   bounds = ((None,None), (0, None)),
                   method = "L-BFGS-B")
        print(reg.message)
    else:
        raise ValueError()
    return (reg.x[0], reg.x[1])


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


class UnderlineRegressionFigure:
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # color[0] = 'blue'
    # color[1] = 'orange'
    # color[2] = 'green'
    # color[3] = 'red'
    def __init__(self, recording, roi_number, method = 'BFGS'):
        sns.set_style("dark")
        sns.set_context("paper")
        robust = method in ('robust','seigel','theil sen')
        f = np.stack((np.zeros(recording.F[roi_number,:].shape),
                      recording.F[roi_number,:], 
                      recording.Fneu[roi_number,:]
                              ),axis=-1)
        if method in ("BFGS","grad descent", "gradient descent"):
            alt_theta = underline_regression(f[:,2], f[:,1],method="BFGS")
            alt_f = subtract_bg(f[:,1], f[:,2], alt_theta)
            alt_df_f = get_df_on_f0(alt_f)
        elif method in ("cg","CG"):
            alt_theta = underline_regression(f[:,2], f[:,1],method="CG")
            alt_f = subtract_bg(f[:,1], f[:,2], alt_theta)
            alt_df_f = get_df_on_f0(alt_f)
        else:
            alt_theta = underline_regression(f[:,2], f[:,1],method=method)
            alt_f = subtract_bg(f[:,1], f[:,2], alt_theta)
            alt_df_f = get_df_on_f0(alt_f)
            
        
        underline_theta = underline_regression(f[:,2], f[:,1])
        underline_f = subtract_bg(f[:,1], f[:,2], underline_theta)
        underline_df_f = get_df_on_f0(underline_f)
        
        self.fig = plt.figure(figsize=(8,4),tight_layout=True)
        
        regression_axis = self.fig.add_axes((0.075,0.1,0.4,0.85))
        regression_axis.plot(f[:,2], f[:,1], 'o', color = self.color[0])

        regression_axis.plot(f[:,2], alt_theta[0] + alt_theta[1]*f[:,2], label = f'Underline Regression with {method} method',
                             color = self.color[1])
        regression_axis.plot(f[:,2], underline_theta[0] + underline_theta[1]*f[:,2], label = 'Underline Regression with Powell method',
                             color = self.color[2])
        regression_axis.set_ylabel('Cell Fluorescence')
        regression_axis.set_xlabel('Neuropil Fluorescence')
        regression_axis.legend(loc='upper left')

        
        alt_axis = self.fig.add_axes((0.55,0.5,0.4,0.4))

        alt_axis.plot(alt_df_f, label = f'Underline Regression with {method} Method',
                      color = self.color[1])
        alt_axis.set_ylabel('DF/F')
        alt_axis.set_xticks([])
        alt_axis.legend(loc='upper right')
        underline_axis = self.fig.add_axes((0.55,0.1,0.4,0.4))
        underline_axis.plot(underline_df_f, label = 'Underline Regression with Powell Method',
                            color=self.color[2])
        underline_axis.set_ylabel('DF/F')
        underline_axis.set_xlabel('Sample')
        underline_axis.set_xticks([])
        underline_axis.legend(loc='upper right')
        
        self.axes = (regression_axis, alt_axis, underline_axis)
        self.f = f
    def show(self):
        self.fig.show()


class ThreeKindsOfRegressionFigure(UnderlineRegressionFigure):
    def __init__(self,recording,roi_number):
        super().__init__(recording,roi_number)
        (regression_axis, alt_axis, underline_axis) = self.axes
        f = self.f
        alt_axis.set_position((0.55,0.37,0.4,0.27),which='original')
        underline_axis.set_position((0.55,0.1,0.4,0.27),which = 'original')
        
        boring_theta = guess(f[:,2], f[:,1])
        boring_f = subtract_bg(f[:,1], f[:,2], boring_theta)
        boring_df_f = get_df_on_f0(boring_f)
        regression_axis.plot(f[:,2], boring_theta[0] + boring_theta[1]*f[:,2], 
                             label = 'OLS Regression',
                             color = self.color[3])
        regression_axis.legend()
        boring_axis = self.fig.add_axes((0.55,0.64,0.4,0.27))
        boring_axis.plot(boring_df_f, label = 'OLS Regresison', color = self.color[3])
        boring_axis.set_ylabel('DF/F')
        boring_axis.set_xticks([])
        boring_axis.legend(loc='upper right')


class DescentFigure:
    def __init__(self,recording = rec, i = 4):
        self.f = np.stack((np.zeros(recording.F[i,:].shape),
          recording.F[i,:], 
          recording.Fneu[i,:]
                  ),axis=-1)
        self.X = rec.Fneu[i]
        self.Y = rec.F[i]
        # self.attempts = list(reversed(self.regress().allvecs))
        # self.fig,ax = plt.subplots()
        # ax.plot(self.f[:,2], self.f[:,1], 'o')
        # self.line, = ax.plot([],[])
        # self.animation = FuncAnimation(self.fig,self.update_line,
        #                                len(self.attempts), blit=True,
        #                                interval = 500,
        #                                repeat_delay = 1000)

    def regress(self):
        x,y = self.X, self.Y
        start_params = guess(x, y)
        reg = minimize(qloss,
                   x0 = start_params,
                   jac = qgrad,
                   args = (x, y),
                   bounds = ((None, None), (0, None)),
                   method = "L-BFGS-B")
        return reg

plt.close('all')


ThreeKindsOfRegressionFigure(rec,4).show()