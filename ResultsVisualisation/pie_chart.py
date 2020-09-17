# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:10:20 2020

@author: viviani
"""
from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

class PieChartAnovaFigure:
    colors = sns.color_palette()
    def __init__(self,df,dataset='left_only',statistic='f'):
        pvalue_cols = [c for c in df.columns if 'pvalue' in c and 'ANOVA' in c]
        fvalue_cols = [c for c in df.columns if 'fvalue' in c]
        eta_cols    = [c for c in df.columns if 'partial_eta2' in c]
        pvals = df[pvalue_cols]
        fvals = df[fvalue_cols]
        evals = df[eta_cols]
        print(evals.columns)
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
            print('check1')
            for i,c in enumerate(evals.columns):
                print('loop')
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

df1 = pd.read_csv("C:/Users/viviani/Documents/results_left_only.csv")
PieChartAnovaFigure(df1,'left_only','eta').show()
df2 = pd.read_csv("C:/Users/viviani/Documents/results_binocular.csv")
PieChartAnovaFigure(df2,'both_sides','eta').show()
df3 = pd.read_csv("C:/Users/viviani/Documents/results_low_contrast.csv")
PieChartAnovaFigure(df3,'low_contrast','eta').show()

# pvals = df[["trial.segment.pvalue",
#             "trial.segment:correct.pvalue",
#             "trial.segment:go.pvalue"]]
# fvals = df[["trial.segment.fvalue",
#             "trial.segment:correct.fvalue",
#             "trial.segment:go.fvalue"]]

# counter = count_unique_index((pvals<0.05),["trial.segment.pvalue",
#                                            "trial.segment:correct.pvalue",
#                                            "trial.segment:go.pvalue"])
# counter['percent']=counter['count']/counter['count'].sum()

# names = np.array(('Trials','Correct','Go'))
# names = [names[boolrow[:-2].to_numpy().astype('bool')] for _,boolrow in counter.iterrows()]
# names = ['&'.join(ls) if list(ls) else "None" for ls in names]
# fig = plt.figure(figsize = [12,5], tight_layout=True)
# gs = GridSpec(3,2,figure=fig)
# ax0 = fig.add_subplot(gs[:,0])
# ax1 = fig.add_subplot(gs[0,1])
# ax2 = fig.add_subplot(gs[1,1])
# ax3 = fig.add_subplot(gs[2,1])
# ax2.set_ylabel("Frequency")
# ax3.set_xlabel("F value")


# colors = sns.color_palette()



# wedges,text1,text2 = ax0.pie(counter['count'],#labels=names,
#                         autopct='%1.f%%',counterclock=True,
#                         startangle=60, pctdistance=0.9)

# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# kw = dict(arrowprops=dict(arrowstyle="-",color='black'),
#           zorder=5, va="center")

# for i, (p,c) in enumerate(zip(wedges,counter['count'])):
#     ang = (p.theta2 - p.theta1)/2. + p.theta1
#     y = np.sin(np.deg2rad(ang))
#     x = np.cos(np.deg2rad(ang))
#     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
#     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
#     kw["arrowprops"].update({"connectionstyle": connectionstyle})

#     ax0.annotate(names[i], xy=(x, y), xytext=(x+np.sign(x)*0.2,
#                                               y*1.2),
#                  annotation_clip=False,
#                  horizontalalignment=horizontalalignment, **kw)


# ax0.set_title("Significance of predictors on ROI Fluorescence")
# ax1.set_title("F values of predictors (when significant)")
# ax1.hist(fvals["trial.segment.fvalue"][pvals["trial.segment.pvalue"]<0.05],
#           color = colors[4],label='Trial occuring')
# ax2.hist(fvals["trial.segment:correct.fvalue"][pvals["trial.segment:correct.pvalue"]<0.05],
#           color = colors[2], label = 'Correct/Incorrect trial')
# ax3.hist(fvals["trial.segment:go.fvalue"][pvals["trial.segment:go.pvalue"]<0.05],
#           color = colors[1], label = 'Go/No-Go trial')
# xlims = map(lambda a:a.get_xlim(),(ax1,ax2,ax3))
# new_xlim = (0,None)
# for a in (ax1,ax2,ax3): a.set_xlim(new_xlim); a.legend(loc='upper right')
# fig.show()