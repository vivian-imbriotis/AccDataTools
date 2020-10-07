# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:48:47 2020

@author: viviani
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

import rpy2.robjects as R


sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"]   = 11

class SingleAxonVsLickingFigure:
    pass

class LickingCorrelationHistogram:
    palette = sns.color_palette()
    blue = palette[0]
    red  = palette[3]
    grey = palette[7]
    def __init__(self,df):
        nontrial = df[pd.isnull(df.Trial_ID)] #NOT during trials
        rois = nontrial.ROI_ID.unique()
        pearsons_rs = np.zeros(rois.shape)
        pearsons_pvals = np.zeros(rois.shape)
        for idx,roi in enumerate(rois):
            logged_df = nontrial[nontrial.ROI_ID==roi].logged_dF
            licking   = nontrial[nontrial.ROI_ID==roi].lick_factor==0
            r,p = pearsonr(logged_df,licking)
            pearsons_rs[idx]   = r
            pearsons_pvals[idx] = p
        fig,ax = plt.subplots()
        R.globalenv["pvalues"] = R.FloatVector(pearsons_pvals)
        adj_pvals = np.array(R.r("p.adjust(pvalues,method='fdr')"))
        upper_significant = pearsons_rs[(adj_pvals<0.05) * (pearsons_rs>0)]
        lower_significant = pearsons_rs[(adj_pvals<0.05) * (pearsons_rs<0)]
        not_significant   = pearsons_rs[adj_pvals>=0.05]
        s_min, s_max = np.min(lower_significant), np.max(upper_significant)
        ns_min, ns_max = np.min(not_significant), np.max(not_significant)
        ns_num_bins = int(40 * (ns_max-ns_min)/(s_max-s_min))
        step = (ns_max-ns_min)/ns_num_bins
        ns_bin_boundaries = np.linspace(ns_min,
                                        ns_max,
                                        ns_num_bins)
        us_bin_boundaries = np.arange(ns_max,s_max,step)
        ls_bin_boundaries = np.arange(ns_min,s_min,-1*step)[::-1]
        ax.hist(not_significant, bins = ns_bin_boundaries,
                color=self.grey,
                label = "Not significant",
                weights = np.ones(len(not_significant))/len(pearsons_rs))
        ax.hist(upper_significant, bins = us_bin_boundaries,
                color=self.blue,
                label = "Positiviely correlated (p<0.05)",
                weights = np.ones(len(upper_significant))/len(pearsons_rs))
        ax.hist(lower_significant, bins = ls_bin_boundaries,
                color=self.red,
                label = "Negatively correlated (p<0.05)",
                weights = np.ones(len(lower_significant))/len(pearsons_rs))
        ax.legend(loc="upper right")
        ax.set_ylabel("Proportion of ROIs")
        ax.set_xlabel("Correlation between licking and fluorescence (Pearson's R)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.show()
        
if __name__=="__main__":
    df = pd.read_csv(r"C:\Users\viviani\Desktop\single_experiments_for_testing\2016-11-01_03_CFEB027.csv")
    LickingCorrelationHistogram(df)
            
