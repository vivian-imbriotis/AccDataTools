# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:00:50 2020

@author: Vivian Imbriotis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

class LinearRegressionAssumptionsFigure:
    def __init__(self,df,model = ""):
        self.fig,ax = plt.subplots(nrows=2,
                                   tight_layout=True,
                                   figsize = [8,6])
        ax[0].set_title('$\\bf{(A)}$',loc='left')
        ax[0].set_ylabel(f"Proportion of{f' {model}' if model else ''} models")
        ax[0].set_xlabel("Durbin-Watson Statistic")
        ax[0].hist(df.durbin_statistic.values,
                   bins = np.linspace(0,4,30,
                                      endpoint=True),
                   weights = np.ones(len(df))/len(df))
        ax[0].yaxis.set_major_formatter(PercentFormatter(1))
        ax[1].set_title('$\\bf{(B)}$',loc='left')
        ax[1].set_ylabel(f"Proportion of{f' {model}' if model else ''} models")
        ax[1].set_xlabel("Shapiro-Wilks Statistic")
        ax[1].hist(df.shapiro_statistic.values,
                   bins = np.linspace(0,1,30,
                                      endpoint=True),
                   weights = np.ones(len(df))/len(df))
        ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    def show(self):
        self.fig.show()

def read_in_data(path):
    df = pd.read_csv(path)
    full_model_assumption_stats = df[[c for c in df.columns if 'full.model' in c]]
    collapsed_model_assumption_stats = df[[c for c in df.columns if 'collapsed' in c]]
    colnames = ('shapiro_statistic','shapiro_pval',
                'durbin_statistic','durbin_pval',
                'adj_r_squared')
    full_model_assumption_stats.columns = colnames
    collapsed_model_assumption_stats.columns = colnames
    return (full_model_assumption_stats,
            collapsed_model_assumption_stats)

if __name__=="__main__":
    plt.close('all')
    df1,df2 = read_in_data(path = "../RScripts/lin_regress_assumption_statistics.csv")
    LinearRegressionAssumptionsFigure(df1,model='kernel').show()
    LinearRegressionAssumptionsFigure(df2,model='collapsed').show()


    