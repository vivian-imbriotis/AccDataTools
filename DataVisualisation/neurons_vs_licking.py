# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:48:47 2020

@author: viviani
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
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
            logged_df = nontrial[nontrial.ROI_ID==roi].dF_on_F
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
        ns_num_bins = int(100 * (ns_max-ns_min)/(s_max-s_min))
        step = (ns_max-ns_min)/ns_num_bins
        ax.hist(not_significant, bins = np.linspace(ns_min,
                                                    ns_max,
                                                    ns_num_bins),
                color=self.grey)
        ax.hist(upper_significant, bins = np.arange(ns_max,s_max,step),
                color=self.blue)
        ax.hist(lower_significant, bins = np.arange(s_min,ns_min,step),
                color=self.red)
        fig.show()
        
if __name__=="__main__":
    df = pd.read_csv(r"C:\Users\viviani\Desktop\single_experiments_for_testing\2016-11-01_03_CFEB027.csv")
    LickingCorrelationHistogram(df)
            
