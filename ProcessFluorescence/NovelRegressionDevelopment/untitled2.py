# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:56:01 2020

@author: Vivian Imbriotis
"""


from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d
from sklearn.linear_model import TheilSenRegressor, HuberRegressor
from df_on_f_novel_regression_scratchpad import underline_regression
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11


def make_data(n_points = 10000):
    spikes = make_spikes(n_points)
    bg = make_bg(n_points)
    data = {}
    data["frame_n"] = np.arange(0, n_points)
    data["ground_truth"] = spikes
    data["bg"] = bg
    data["slope"] = np.random.random()+0.5
    data["raw"] = data["ground_truth"] + data["slope"]*bg + 5*np.random.rand(n_points)
    return data

def make_spikes(n_points = 10000):
    spike_prob = 0.01
    spike_size_mean = 1
    kernel = make_kernel()
    spike_times = 1.0 * (np.random.rand(n_points-kernel.size+1) < spike_prob)
    spike_amps = np.random.lognormal(mean = spike_size_mean, sigma = 0.5, size=spike_times.size) * spike_times
    spikes = 5*np.convolve(spike_amps, kernel) + 8 + 6*np.random.rand(n_points)
    return spikes

def make_bg(n_points = 10000):
    return 6*(np.sin(np.arange(0, n_points)/(6.28))/2) + 30 + 10*np.random.rand(n_points)


def make_kernel(tau = 10):
    length = tau*8
    time = np.arange(0, length)
    return np.exp(-time/tau)
   

#Just plotting and helper functions below


def plot_data_creation_process(colors = sns.color_palette()):
    data = make_data(1000)
    data = proc_data(data)
    
    #construct plot layout with row titles:
    nrows = 5
    ncols = 2
    fig, big_axes = plt.subplots(nrows = nrows, ncols = 1,tight_layout= True,
                           figsize = (12,9))
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.axis("off")
        big_ax._frameon = False
    ax = np.empty((len(big_axes),ncols),dtype=object)
    for row in range(len(big_axes)):
        ax[row][0] = fig.add_subplot(len(big_axes),ncols,ncols*row+1)
        ax[row][1] = fig.add_subplot(len(big_axes),ncols,ncols*row+2)
    
    for axis in ax.flatten():
        axis.set_xticklabels([])
        

    for axis in ax[2:,0]:
        axis.set_yticks([])
        axis.set_xlabel("Fneu")
        axis.set_ylabel("F")
    big_axes[0].set_title("Ground Truth")
    ax[0][0].set_title("Cell Fluorescence")
    ax[0][0].plot(data["frame_n"], data["ground_truth"])
    ax[0][1].set_title("dF/F0")
    ax[0][1].plot(data["frame_n"], data["gt_df_on_f"])
    big_axes[1].set_title("Background Contamination")
    ax[1][0].set_title("Neuropil Fluorescence")
    ax[1][0].plot(data["frame_n"], data["bg"])
    ax[1][1].set_title("Measured Fluorescence")
    ax[1][1].plot(data["frame_n"], data["raw"])
    
    
    big_axes[2].set_title("Translated Robust Regression (Pseudo-Huber Estimated)")
    ax[2][0].plot(data["bg"],data["raw"], 'o')
    ax[2][0].plot(data["bg"], data["H_theta"][0] + data["H_theta"][1]*data["bg"])
    ax[2][1].plot(data["frame_n"],get_df_on_f0(data["raw"]-data["H_theta"][1]*data["bg"]))
    
    big_axes[3].set_title("Asymmetric L1 norm regression with Powell method")
    ax[3][0].plot(data["bg"],data["raw"], 'o')
    ax[3][0].plot(data["bg"], data["CR_theta"][0] + data["CR_theta"][1]*data["bg"])
    ax[3][1].plot(data["frame_n"],get_df_on_f0(data["raw"]-data["CR_theta"][1]*data["bg"]))

    big_axes[4].set_title("Asymmetric L2 norm regression with L-BFGS-B method")
    ax[4][0].plot(data["bg"],data["raw"], 'o')
    ax[4][0].plot(data["bg"], data["PR_theta"][0] + data["PR_theta"][1]*data["bg"], label = 'Robust Regression')
    ax[4][1].plot(data["frame_n"],get_df_on_f0(data["raw"]-data["PR_theta"][1]*data["bg"]))
    
    
    for axis in ax[0:2,:].flatten():
        axis.set_ylim((-0.5,None))
    fig.show()

def plot_data_creation_process(colors = sns.color_palette()):
    data = make_data(1000)
    data = proc_data(data)
    
    #construct plot layout with row titles:
    nrows = 3
    ncols = 2
    fig, big_axes = plt.subplots(nrows = nrows, ncols = 1,tight_layout= True,
                           figsize = (10,12))
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.axis("off")
        big_ax._frameon = False
    ax = np.empty((len(big_axes),ncols),dtype=object)
    for row in range(len(big_axes)):
        ax[row][0] = fig.add_subplot(len(big_axes),ncols,ncols*row+1)
        ax[row][1] = fig.add_subplot(len(big_axes),ncols,ncols*row+2)
    
    for idx, axis in enumerate(ax.flatten()):
        axis.set_xticklabels([])
        if idx!=4:
            axis.set_xlabel("Time")
        if idx in (0,2,3,5):
            axis.set_ylabel("Fluorescence (AU)")
        elif idx==1:
            axis.set_ylabel("Fluorescence ($\Delta$F/F0 units)")
        
    for axis in ax[2:,0]:
        axis.set_yticks([])
        axis.set_xlabel("Neuropil Fluorescence")
        axis.set_ylabel("Measured Fluorescence")
    
    big_axes[0].set_title("$\\bf{(A)}$ Ground Truth\n\n")
    ax[0][0].set_title("True Cell Fluorescence")
    ax[0][0].plot(data["frame_n"], data["ground_truth"])
    ax[0][1].set_title("True cell $\\Delta$F/F0")
    ax[0][1].plot(data["frame_n"], data["gt_df_on_f"])
    
    big_axes[1].set_title("$\\bf{(B)}$ Background Contamination\n\n")
    ax[1][0].set_title("Neuropil Fluorescence")
    ax[1][0].plot(data["frame_n"], data["bg"])
    ax[1][1].set_title("Measured Cell Fluorescence\n(Cell Fluorecence + $\\beta\\times$Neuropil Fluorescence + $\\epsilon$)")
    ax[1][1].plot(data["frame_n"], data["raw"])
    
    
    big_axes[2].set_title("$\\bf{(C)}$ Neuropil Subtraction with Underline Regression\n\n")
    ax[2][0].set_title("    Underline Regression used to infer...")
    ax[2][0].plot(data["bg"],data["raw"], 'o')
    ax[2][0].plot(data["bg"], data["H_theta"][0] + data["H_theta"][1]*data["bg"])
    ax[2][1].set_title("...an estimation of true cell fluorescence")
    ax[2][1].plot(data["frame_n"],(data["raw"]-data["H_theta"][1]*data["bg"]))
    
    
    
    for axis in ax[0:2,:].flatten():
        axis.set_ylim((-0.5,None))
    fig.show()

def plot_data():
    data = make_data()
    data = proc_data(data)
    fig, ax = plt.subplots(nrows = 3, ncols = 2,figsize = (15, 9))

    ax[0][0].plot(data["frame_n"], data["raw"])
    ax[0][0].set_ylabel('Raw F')
    ax[0][0].set_xlabel('Frame Number')

    ax[0][1].plot(data["bg"], data["raw"], 'o')
    ax[0][1].plot(data["bg"], data["NRR_theta"][0] + data["NRR_theta"][1]*data["bg"], label = 'Naive Robust Regression')
    ax[0][1].plot(data["bg"], data["RR_theta"][0] + data["RR_theta"][1]*data["bg"], label = 'Robust Regression')
    ax[0][1].set_ylabel('Total Fluorescence')
    ax[0][1].set_xlabel('Neuropil Fluorescence')
    ax[0][1].legend(loc='upper left')

    ax[1][0].plot(data["frame_n"], data["ground_truth"])
    ax[1][0].set_ylabel("Ground Truth")
    

    ax[1][1].plot(data["frame_n"], data["NRR_df_on_f"])
    ax[1][1].set_ylabel("Naive Robust Reg df/f")

    ax[2][0].plot(data["frame_n"], data["RR_df_on_f"])
    ax[2][0].set_ylabel("Robust Regression df/f")

    ax[2][1].plot(data["ground_truth"], data["RR_df_on_f"], 'o')
    ax[2][1].plot(data["ground_truth"], data["ground_truth"], label= 'Unity')
    ax[2][1].set_title("MeanSquareError = " + str(calc_error(data,df=True)))
    ax[2][1].set_xlabel("Ground Truth")
    ax[2][1].set_ylabel("Robust Regression Aproach")
    ax[2][1].legend(loc='upper left')
    fig.show()

def compare_approaches(df = False):
    data = make_data()
    data = proc_data(data)
    fig, ax = plt.subplots(nrows = 4, ncols = 2,figsize = (15, 12),
                           tight_layout=True)
    
    ax[0][0].plot(data["frame_n"], data["gt_df_on_f"]if df else data["ground_truth"])
    ax[0][0].set_ylabel(f"Ground Truth {'dF/F0' if df else 'Fcell'}")

    ax[0][1].plot(data["bg"], data["raw"], 'o')
    ax[0][1].plot(data["bg"], data["RR_theta"][0] + data["RR_theta"][1]*data["bg"], label = 'Robust Regression')
    ax[0][1].plot(data["bg"], data["CR_theta"][0] + data["CR_theta"][1]*data["bg"], label = 'Biased L1 Regression (Powell Method)')
    ax[0][1].plot(data["bg"], data["PR_theta"][0] + data["PR_theta"][1]*data["bg"], label = 'Biased L2 Regression (BFGS Method)')
    ax[0][1].set_ylabel('Total Fluorescence')
    ax[0][1].set_xlabel('Neuropil Fluorescence')
    ax[0][1].legend(loc='upper left')


    ax[1][1].plot(data["gt_df_on_f"]if df else data["ground_truth"], 
                  data["RR_df_on_f"] if df else data["RR_bg_subtracted"], 'o')
    ax[1][1].plot(data["gt_df_on_f"]if df else data["ground_truth"], 
                  data["gt_df_on_f"]if df else data["ground_truth"], 
                  label= 'Unity')
    ax[1][1].set_title(f"MeanSquareError = {calc_error(data,df)[0]:.8f}")
    ax[1][1].set_xlabel("Ground Truth")
    ax[1][1].set_ylabel("Robust Regression Approach")
    ax[1][1].legend(loc='upper left')

    ax[1][0].plot(data["frame_n"], 
                  data["RR_df_on_f"] if df else data["RR_bg_subtracted"])
    ax[1][0].set_ylabel(f"Robust Regression {'dF/F0' if df else 'Fcell'}")

    ax[2][0].plot(data["frame_n"], 
                  data["CR_df_on_f"] if df else data["CR_bg_subtracted"])
    ax[2][0].set_ylabel(f"L1-Heaviside Regression {'dF/F0' if df else 'Fcell'}")

    ax[2][1].plot(data["gt_df_on_f"]if df else data["ground_truth"], 
                  data["CR_df_on_f"] if df else data["CR_bg_subtracted"], 'o')
    ax[2][1].plot(data["gt_df_on_f"]if df else data["ground_truth"], 
                  data["gt_df_on_f"]if df else data["ground_truth"], 
                  label= 'Unity')
    ax[2][1].set_title(f"MeanSquareError = {calc_error(data,df)[1]:.8f}")
    ax[2][1].set_xlabel("Ground Truth")
    ax[2][1].set_ylabel("Custom Regression Approach 1")
    ax[2][1].legend(loc='upper left')

    ax[3][0].plot(data["frame_n"], 
                  data["PR_df_on_f"] if df else data["PR_bg_subtracted"])
    ax[3][0].set_ylabel(f"Biased L2 vs L2k norm Regression {'dF/F0' if df else 'Fcell'}")

    ax[3][1].plot(data["gt_df_on_f"]if df else data["ground_truth"], 
                  data["PR_df_on_f"] if df else data["PR_bg_subtracted"], 'o')
    ax[3][1].plot(data["gt_df_on_f"]if df else data["ground_truth"], 
                  data["gt_df_on_f"]if df else data["ground_truth"], 
                  label= 'Unity')
    ax[3][1].set_title(f"MeanSquareError = {calc_error(data,df)[2]:.8f}")
    ax[3][1].set_xlabel("Ground Truth")
    ax[3][1].set_ylabel("Custom Regression Approach 2")
    ax[3][1].legend(loc='upper left')
    fig.show()


def time_approaches():
    N = np.linspace(80,100000,10).astype(int)
    robust_regression_times = np.zeros(N.shape)
    powell_times = np.zeros(N.shape)
    L2_norm_times = np.zeros(N.shape)
    for idx,n_points in enumerate(N):
        print(idx)
        t1=np.zeros(10); t2=np.zeros(10); t3=np.zeros(10); t4=np.zeros(10)
        for i in range(1):
            data = make_data(n_points)
            t1[i] = time()
            robust_regression(data["bg"], data["raw"])
            t2[i] = time()
            custom_regression(data["bg"], data["raw"])
            t3[i] = time()
            parabolic_regression(data["bg"], data["raw"])
            t4[i] = time()
        robust_regression_times[idx] = (t2-t1).mean()
        powell_times[idx] = (t3-t2).mean()
        L2_norm_times[idx] = (t4-t3).mean()
    fig,ax = plt.subplots(nrows = 3, sharex=True)
    ax[0].set_title("Robust Regression Time Complexity")
    ax[0].plot(N,robust_regression_times)
    ax[1].set_title("Powell Method Time Complexity")
    ax[1].plot(N,powell_times)
    ax[1].set_ylabel("Execution Time")
    ax[2].set_title("BFGS Method Time Complexity")
    ax[2].plot(N,L2_norm_times)
    ax[2].set_xlabel("Number of data points")
    fig.show()

        

def proc_data(data, do_all=True):
    data["gt_df_on_f"] = get_df_on_f0(data["ground_truth"])
    if do_all:
        data["NRR_theta"] = naive_robust_regression(data["bg"], data["raw"])
        data["NRR_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["NRR_theta"])
        data["NRR_df_on_f"] = get_df_on_f0(data["NRR_bg_subtracted"])
    
    t1 = time()
    data["RR_theta"] = robust_regression(data["bg"], data["raw"])
    data["RR_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["RR_theta"])
    data["RR_df_on_f"] = get_df_on_f0(data["RR_bg_subtracted"])
    data["RR_RMSE"] = np.sum((data["gt_df_on_f"] - data["RR_df_on_f"])**2)**0.5
    data["RR_time"] = time() - t1
    
    t2 = time()
    data["TH_theta"] = translated_huber_regression(data["bg"], data["raw"])
    data["TH_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["TH_theta"])
    data["TH_df_on_f"] = get_df_on_f0(data["TH_bg_subtracted"])
    data["TH_RMSE"] = np.sum((data["gt_df_on_f"] - data["TH_df_on_f"])**2)**0.5
    data["TH_time"] = time() - t2
    
    t3 = time()
    data["CR_theta"] = custom_regression(data["bg"], data["raw"])
    data["CR_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["CR_theta"])
    data["CR_df_on_f"] = get_df_on_f0(data["CR_bg_subtracted"])
    data["CR_RMSE"] = np.sum((data["gt_df_on_f"] - data["CR_df_on_f"])**2)**0.5
    data["CR_time"] = time() - t3

    t4 = time()
    data["PR_theta"] = parabolic_regression(data["bg"], data["raw"])
    data["PR_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["PR_theta"])
    data["PR_df_on_f"] = get_df_on_f0(data["PR_bg_subtracted"])
    data["PR_RMSE"] = np.sum((data["gt_df_on_f"] - data["PR_df_on_f"])**2)**0.5
    data["PR_time"] = time() - t4

    t5 = time()
    data["H_theta"] = huber_regression(data["bg"], data["raw"])
    data["H_bg_subtracted"] = subtract_bg(data["raw"], data["bg"], data["H_theta"])
    data["H_df_on_f"] = get_df_on_f0(data["H_bg_subtracted"])
    data["H_RMSE"] = np.sum((data["gt_df_on_f"] - data["H_df_on_f"])**2)**0.5
    data["H_time"] = time() - t5
    return data

def calc_error(data,df):
    ground_truth = data["gt_df_on_f"]if df else data["ground_truth"]
    return (np.sum((ground_truth - (data["RR_df_on_f"] if df else data["RR_bg_subtracted"]))**2)/ground_truth.size,
            np.sum((ground_truth - (data["CR_df_on_f"] if df else data["CR_bg_subtracted"]))**2)/ground_truth.size,
            np.sum((ground_truth - (data["PR_df_on_f"] if df else data["PR_bg_subtracted"]))**2)/ground_truth.size)

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

    # subtracted_data = subtract_bg(y, x, [reg.coef_[0], reg.coef_[1]])
    subtracted_data = y - reg.coef_[0] - reg.coef_[1]*x
    offset = np.min(subtracted_data)

    return np.array([reg.coef_[0]+offset, reg.coef_[1]])

def translated_huber_regression(x,y):
    y_reshape = y.reshape(-1, 1)
    X = np.vstack((np.ones(y_reshape.shape).transpose(), x.reshape(-1, 1).transpose()))
    reg = TheilSenRegressor(random_state=0).fit(X.transpose(), np.ravel(y_reshape))

    # subtracted_data = subtract_bg(y, x, [reg.coef_[0], reg.coef_[1]])
    subtracted_data = y - reg.coef_[0] - reg.coef_[1]*x
    offset = np.min(subtracted_data)
    return np.array([reg.coef_[0]+offset, reg.coef_[1]])

def huber_regression(x,y):
    return underline_regression(x,y,method = 'huber')

def custom_regression(x, y):
    return underline_regression(x,y, method = "ramp")

def parabolic_regression(x,y):
    return underline_regression(x,y,method="parabolic")

def subtract_bg(f, bg, theta):
    # return f - ( theta[0] + theta[1] * bg)
    return f - theta[1]*bg

def get_smoothed_running_minimum(timeseries, tau1 = 30, tau2 = 100):
    result = minimum_filter1d(uniform_filter1d(timeseries,tau1,mode='nearest'),
                            tau2,
                            mode = 'reflect')
    return result

def get_df_on_f0(F,F0=None):
    if not F0 is None:
        return (F - F0) / F0
    else:
        F0 = get_smoothed_running_minimum(F)
        return get_df_on_f0(F,F0)

def run_simulation():
    RR = []
    TH = []
    CR = []
    PR = []
    H = []
    RR_times = []
    TH_times = []
    CR_times = []
    PR_times = []
    H_times = []
    for i in range(100):
        data = make_data()
        data = proc_data(data,do_all=False)
        RR.append(data["RR_RMSE"])
        TH.append(data["TH_RMSE"])
        CR.append(data["CR_RMSE"])
        PR.append(data["PR_RMSE"])
        H.append(data["H_RMSE"])
        RR_times.append(data["RR_time"])
        TH_times.append(data["TH_time"])
        CR_times.append(data["CR_time"])
        PR_times.append(data["PR_time"])
        H_times.append(data["H_time"])
    fig,ax = plt.subplots(nrows=5, ncols = 2,figsize = (20,12),constrained_layout=True)
    for (ax0,ax1),name,err,time in zip(ax, 
                              ("Translated Theil-Sen",
                               "Translated Huber",
                                "Heaviside/L1 norm with Powell",
                                "Asymmetric L2 norm with BFGS",
                                "L2/PseudoHuber with BFGS"),
                              (RR,TH,CR,PR,H),
                              (RR_times,TH_times,CR_times,PR_times,H_times)):
        ax0.set_title(f"{name} Accuracy")
        ax0.set_ylabel("frequency")
        ax0.set_xlabel("RMSE (df/f units)")
        ax0.hist(err)
        ax1.set_title(f"{name} Timing")
        ax1.set_xlabel("Time taken to fit 10000 data points (s)")
        ax1.set_ylabel("frequency")
        ax1.hist(time)
    fig.show()
    
    return {'TheilSen':RR,
            'Huber':TH,
            'RampL1':CR,
            'AsymmetricL2':PR,
            'L2PseudoHuber':H,
            'TheilSen_times':RR_times,
            'Huber_times':TH_times,
            'RampL1_times':CR_times,
            'AsymmetricL2_times':PR_times,
            'L2PseudoHuber_times':H_times}

# plot_data_creation_process()
if __name__=="__main__":
    plot_data_creation_process()