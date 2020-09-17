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
from adjustText import adjust_text
sns.set_style("darkgrid")


readable_titles = {
    "trial.segment.fvalue" : "During Trial",
    "trial.segment:correct.fvalue" : "Correct/Incorrect Response",
    "trial.segment:go.fvalue" : "Go/No-Go Stimulus",
    "trial.segment:side.fvalue" : "Left/Right Stimulus",
    "trial.segment:correct:contrast.fvalue" : "High/Low Contrast",
    "trial.segment.pvalue" : "During Trial",
    "trial.segment:correct.pvalue" : "Correct/Incorrect Response",
    "trial.segment:go.pvalue" : "Go/No-Go Stimulus",
    "trial.segment:side.pvalue" : "Left/Right Stimulus",
    "trial.segment:correct:contrast.pvalue" : "High/Low Contrast"
    }

def count_unique_index(df, by):                                                                                                                                                 
    return df.groupby(by).size().reset_index().rename(columns={0:'count'})

df = pd.read_csv("C:/Users/viviani/Documents/left_only_collapsed_lm_anova_results.csv")

class PieChartAnovaFigure:
    colors = sns.color_palette()
    def __init__(self,df,dataset='left_only'):
        pvalue_cols = [c for c in df.columns if 'pvalue' in c]
        fvalue_cols = [c for c in df.columns if 'fvalue' in c]
        pvals = df[pvalue_cols]
        fvals = df[fvalue_cols]
        print(pvalue_cols)
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
        for i,c in enumerate(fvals.columns):
            right_ax.append(
                self.fig.add_subplot(gs[i,1])
                )
            right_ax[i].hist(fvals[c][pvals[c.replace('fvalue','pvalue')]<0.05],
                         label = readable_titles[c],
                         color = self.colors[i])
            right_ax[i].legend()
        

        right_ax[2].set_ylabel("Frequency")
        right_ax[-1].set_xlabel("F value")
        
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
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            annotations.append(
                left_ax.annotate(names[i], xy=(x, y), 
                                 xytext=(x+np.sign(x)*0.2,y*1.2),
                                 annotation_clip=False,
                                 horizontalalignment=horizontalalignment, 
                                 **kw)
                )
        left_ax.set_title("Significance of predictors on ROI Fluorescence")
        right_ax[0].set_title("F values of predictors (when significant)")
        new_xlim = (0,None)
        for a in right_ax: a.set_xlim(new_xlim); a.legend(loc='upper right')
        adjust_text(annotations)
    def show(self):
        self.fig.show()

df = pd.read_csv("C:/Users/viviani/Documents/dump.csv")
PieChartAnovaFigure(df,'left_only').show()

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