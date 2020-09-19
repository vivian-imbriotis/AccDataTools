# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:10:20 2020

@author: viviani
"""
from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_style("darkgrid")


readable_titles = {
    "ANOVA trial.segment fvalue" : "During Trial",
    "ANOVA trial.segment:correct fvalue" : "Correct/Incorrect Response",
    "ANOVA trial.segment:go fvalue" : "Go/No-Go Stimulus",
    "ANOVA trial.segment:side fvalue" : "Left/Right Stimulus",
    "ANOVA trial.segment:correct:contrast fvalue" : "High/Low Contrast",
    "ANOVA trial.segment pvalue" : "During Trial",
    "ANOVA trial.segment:correct pvalue" : "Correct/Incorrect Response",
    "ANOVA trial.segment:go pvalue" : "Go/No-Go Stimulus",
    "ANOVA trial.segment:side pvalue" : "Left/Right Stimulus",
    "ANOVA trial.segment:correct:contrast pvalue" : "High/Low Contrast",
    "ANOVA trial.segment partial_eta2" : "During Trial",
    "ANOVA trial.segment:correct partial_eta2" : "Correct/Incorrect Response",
    "ANOVA trial.segment:go partial_eta2" : "Go/No-Go Stimulus",
    "ANOVA trial.segment:side partial_eta2" : "Left/Right Stimulus",
    "ANOVA trial.segment:correct:contrast partial_eta2" : "High/Low Contrast"
    }

def count_unique_index(df, by):                                                                                                                                                 
    return df.groupby(by).size().reset_index().rename(columns={0:'count'})

df = pd.read_csv("C:/Users/viviani/Documents/left_only_collapsed_lm_anova_results.csv")

class CollapsedModelPieChartAnovaFigure:
    colors = sns.color_palette()
    def __init__(self,df,dataset='left_only',statistic='f'):
        pvalue_cols = [c for c in df.columns if 'pvalue' in c and 'ANOVA' in c]
        fvalue_cols = [c for c in df.columns if 'fvalue' in c]
        eta_cols    = [c for c in df.columns if 'partial_eta2' in c]
        pvals = df[pvalue_cols]
        fvals = df[fvalue_cols]
        evals = df[eta_cols]
        if statistic.lower() in ('f','f value','fvalue'):
            mode = 'f'
        elif statistic.lower() in ('e','eta','eta2','eta squared'):
            mode = 'eta2'
        else:
            raise ValueError(f"statistic muse be 'f' or 'eta2', not {statistic}")
        
        counter = count_unique_index((pvals<0.05),pvalue_cols)
        counter['percent']=counter['count']/counter['count'].sum()
        if dataset=='left_only':     names = np.array(('Trials','Correct','Go'))
        elif dataset=='both_sides':  names = np.array(('Trials','Correct','Go',
                                                    'Side'))
        elif dataset=='low_contrast':names  = np.array(('Trials','Correct','Go',
                                                    'Side','Contrast'))
        
        names = [names[boolrow[:-2].to_numpy().astype('bool')] for _,boolrow in counter.iterrows()]
        names = np.array(['&'.join(ls) if list(ls) else "None" for ls in names])
        self.fig = plt.figure(figsize = [12,5], tight_layout=True)
        gs = GridSpec(len(fvals.columns),2,figure=self.fig)
        right_ax = []
        left_ax = self.fig.add_subplot(gs[:,0])
        if mode=='f':
            for i,c in enumerate(fvals.columns):
                right_ax.append(
                    self.fig.add_subplot(gs[i,1])
                    )
                right_ax[i].hist(fvals[c][pvals[c.replace('fvalue','pvalue')]<0.05],
                             label = readable_titles[c],
                             color = self.colors[i])
                right_ax[i].legend()
        elif mode=='eta2':
            for i,c in enumerate(evals.columns):
                right_ax.append(
                    self.fig.add_subplot(gs[i,1])
                    )
                right_ax[i].hist(evals[c][pvals[c.replace('partial_eta2','pvalue')]<0.05],
                             label = readable_titles[c],
                             color = self.colors[i])
                right_ax[i].legend()
        

        right_ax[2].set_ylabel("Frequency")
        right_ax[-1].set_xlabel("F value" if mode=='f' else "Partial $\eta^2$")
        
        wedges,text1,text2 = left_ax.pie(counter['count'],#labels=names,
                                autopct='%1.f%%',counterclock=True,
                                startangle=60, pctdistance=0.9)
        

        kw = dict(arrowprops=dict(arrowstyle="-",color='black'),
                  zorder=5, va="center")
        
        annotations = []
        for i, (p,c) in enumerate(zip(wedges,counter['count'])):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            text_xy = (x*1.2,y*1.2)
            if dataset=='left_only':
                if names[i]=="Trials&Go":
                    text_xy = (text_xy[0]+0.1,text_xy[1]-0.1)
                elif names[i]=="Trials&Correct":
                    text_xy = (text_xy[0]-0.2,text_xy[1]+0.05)
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            annotations.append(
                left_ax.annotate(names[i], xy=(x, y), 
                                 xytext= text_xy,
                                 annotation_clip=False,
                                 horizontalalignment=horizontalalignment, 
                                 **kw)
                )
        left_ax.set_title("Significance of predictors on ROI Fluorescence")
        if mode=='f':
            right_ax[0].set_title("F values of predictors (when significant)")
        elif mode=='eta2':
            right_ax[0].set_title("Partial Eta Squared values of predictors (when significant)")
        new_xlim = (0,None)
        for a in right_ax: a.set_xlim(new_xlim); a.legend(loc='upper right')
    def show(self):
        self.fig.show()

class CollapsedModelCoefficientEstimatesFigure:
    def __init__(self,df):
        colnames = ("Tone Bin","Stimulus Bin","Response Bin")
        coefs = df[[c for c in df1.columns if ('coefficient' in c and 
                                                'estimate' in c and
                                                'lick' not in c)]]
        intercept = coefs["coefficient X.Intercept. estimate"].to_numpy()
        intercept = np.stack([intercept]*3).transpose()
        main_effect = coefs.iloc[:,0:3]
        main_effect.columns = colnames
        main_effect.loc[:,"Tone Bin"] = 0
        correct = coefs[[c for c in coefs.columns if 'correct1' in c]]
        go      =  coefs[[c for c in coefs.columns if 'go1' in c]]
        correct.columns = colnames
        go.columns      = colnames
        self.fig, ax = plt.subplots(ncols = 2, nrows = 2)
        ax[0][0].set_ylabel("Estimated effect ($\Delta$F/F0 units)")
        ax[0][0].set_title("Main Effect of Trial")
        ax[0][0].plot(main_effect.transpose(), color = 'k',
                      alpha=0.05)
        ax[0][1].set_ylabel("Estimated effect ($\Delta$F/F0 units)")
        ax[0][1].set_title("Effect of Go/No-go")
        ax[0][1].plot(go.transpose(),color='k',alpha=0.05)
        
        ax[1][0].set_ylabel("Estimated effect ($\Delta$F/F0 units)")
        ax[1][0].set_title("Effect of Correct/Incorrect")
        ax[1][0].plot(correct.transpose(),color='k',alpha=0.05)
        
        ax[1][1].set_title('Overall Estimates')
        ax[1][1].set_ylabel("Prediction ($\Delta$F/F0 units)")
        hits = (intercept + main_effect + correct + go).transpose()
        misses = (intercept + main_effect + go).transpose()
        crs = (intercept + main_effect + correct).transpose()
        fas = (intercept+main_effect).transpose()
        for i, (hit,miss,cr,fa) in enumerate(zip(hits,misses,crs,fas)):
            ax[1][1].plot(hits[i],color='green',
                          alpha = 0.0125, label = None)
            ax[1][1].plot(misses[i],color='palegreen',
                          alpha = 0.0125, label = None)
            ax[1][1].plot(crs[i],color='red',
                          alpha = 0.0125, label = None)
            ax[1][1].plot(fas[i],color='darksalmon',
                          alpha = 0.0125, label = None)
        patch1 = mpatches.Patch(color='green', label='Hits')
        patch2 = mpatches.Patch(color='red', label='Misses')
        patch3 = mpatches.Patch(color='palegreen', label='False Alarms')
        patch4 = mpatches.Patch(color='darksalmon', label='Correct Rejections')
        ax[1][1].legend(handles = [patch1,patch2,patch3,patch4])
    def show(self):
        self.fig.show()

class LickingModelFigure:
    def __init__(self,df):
        coefs = df[[c for c in df.columns if ('coefficient' in c and 
                                                'pvalue' in c and
                                                'lick' in c)]]
        
        intercept = coefs["lick.coefficient X.Intercept. pvalue"].to_numpy()
        kernels = coefs.iloc[:,1:]
        self.fig, ax = plt.subplots(ncols = 2, figsize = [8,6],
                                    tight_layout=True)

        artist = ax[0].plot(kernels.transpose(),color='k',alpha = 0.05)
        artist[0].set_label('Value for a single ROI')
        ax[0].legend()
        ax[0].set_xlabel("$\Delta$t around a lick")
        ax[0].set_ylabel("Coefficient Value")
        ax[0].set_xticks([0,11,21])
        ax[0].set_xticklabels([-2,0,2])
        ax[1].set_ylabel('Coefficient Value')
        ax[1].set_xlabel("$\Delta$t around a lick")
        ax[1].set_xticks([0,11,21])
        ax[1].set_xticklabels([-2,0,2])
        ax[1].plot(kernels.mean().transpose(),color='k',
                   label = "mean across axons")
        ax[1].legend()
    def show(self):
        self.fig.show()

def print_anova_stats(df):
    anova_pvals = df[[c for c in df.columns if 'ANOVA' in c and 'pvalue' in c]]
    print("NUMBER OF SIGNIFICANT ROIS")
    print(f"total = {len(df)}")
    print((anova_pvals<0.05).sum())
    print("\nPERCENTAGE SIGNIFICANT ROIS")
    print(100*(anova_pvals<0.05).sum()/len(anova_pvals))
    
def read_in_data():
    df1 = pd.read_csv("../RScripts/results_left_only.csv")
    df2 = pd.read_csv("../RScripts/results_binocular.csv")
    df3 = pd.read_csv("../RScripts/results_low_contrast.csv")
    return(df1,df2,df3)



        
def print_all_findings(df1,df2,df3):
    for i,n in zip((df1,df2,df3),('Monocular','Binocular','LowCon')):
        print(n)
        print_anova_stats(i)
        print("\n\n")

if __name__=="__main__":
    plt.close('all')
    df1,df2,df3 = read_in_data()
    print_all_findings(df1,df2,df3)
    CollapsedModelPieChartAnovaFigure(df1,'left_only','eta').show()
    CollapsedModelPieChartAnovaFigure(df2,'both_sides','eta').show()
    CollapsedModelPieChartAnovaFigure(df3,'low_contrast','eta').show()
    CollapsedModelCoefficientEstimatesFigure(df1).show()
    CollapsedModelCoefficientEstimatesFigure(df2).show()
    CollapsedModelCoefficientEstimatesFigure(df3).show()  
    LickingModelFigure(df1).show()
    LickingModelFigure(df2).show()
    LickingModelFigure(df3).show()
    
    