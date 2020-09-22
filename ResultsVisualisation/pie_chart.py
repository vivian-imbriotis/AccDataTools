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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import animation
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
        
        ax[1][1].set_title('Overall Mean Estimates')
        ax[1][1].set_ylabel("Prediction ($\Delta$F/F0 units)")
        hits = (intercept + main_effect + correct + go).mean()
        misses = (intercept + main_effect + go).mean()
        crs = (intercept + main_effect + correct).mean()
        fas = (intercept+main_effect).mean()
        ax[1][1].plot(hits,color='green',
                      alpha = 1, label = 'Hits')
        ax[1][1].plot(misses,color='palegreen',
                      alpha = 1, label = 'Misses')
        ax[1][1].plot(crs,color='red',
                      alpha = 1, label = 'Correct Rejections')
        ax[1][1].plot(fas,color='darksalmon',
                      alpha = 1, label = 'False Alarms')
        ax[1][1].legend()
    def show(self):
        self.fig.show()

a = None
class LickingModelFigure:
    def __init__(self,df):
        coefs = df[[c for c in df.columns if ('coefficient' in c and 
                                                'pvalue' in c and
                                                'lick' in c)]]
        
        intercept = coefs["lick.coefficient X.Intercept. pvalue"].to_numpy()
        kernels = coefs.iloc[:,1:]
        self.fig1, ax = plt.subplots(ncols = 2, figsize = [8,6],
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
        # self.fig2, pca_ax = plt.subplots(ncols=3)
        pca = PCA(n_components=2)
        kernels_in_pca_coords = pca.fit_transform(kernels.to_numpy())
        # pc1, pc2 = pca.components_
        # pca_ax[2].plot(kernels_in_pca_coords[:,0],kernels_in_pca_coords[:,1],'o')
        # pca_ax[2].set_ylabel("Second Pricipal Component")
        # pca_ax[2].set_xlabel("First Principal Component")
        # pca_ax[0].plot(pc1)
        # pca_ax[0].set_title("First Principal Component")
        # pca_ax[1].plot(pc2)
        # pca_ax[1].set_title("Second Principal Component")
        pca = PCA(n_components = 3)
        kernels_in_pcs = pca.fit_transform(kernels.to_numpy())
        pc1,pc2,pc3 = pca.components_
        self.fig3, pca_ax = plt.subplots(ncols=3)
        pca_ax[0].plot(pc1)
        pca_ax[0].set_title("First Principal Component")
        pca_ax[1].plot(pc2)
        pca_ax[1].set_title("Second Principal Component")
        pca_ax[2].set_title("Third Principal Component")
        pca_ax[2].plot(pc3)
        self.fig4 = plt.figure()
        self.ax3d = self.fig4.add_subplot(111, projection='3d')
        self.ax3d.scatter(kernels_in_pcs[:,0],kernels_in_pcs[:,1],kernels_in_pcs[:,2],
                          s = 1)
        self.ax3d.set_xlabel("Component 1")
        self.ax3d.set_ylabel("Component 2")
        self.ax3d.set_zlabel("Component 3")
        

    def rotate(self,angle):
         self.ax3d.view_init(azim=angle)
    def save(self,name):
        angle = 3
        ani = animation.FuncAnimation(self.fig4, self.rotate, frames=np.arange(0, 360, angle), interval=50, repeat = True)
        ani.save(f'{name}.gif', writer=animation.PillowWriter(fps=20))
    def show(self):
        self.fig1.show()
        self.fig3.show()
        self.fig4.show()

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


class SubtypedROIsWithSignificantTrialResponseFigure:
    def __init__(self):
            coefs = df[[c for c in df.columns if ('ANOVA' in c and 
                                            'pvalue' in c)]]

        
def print_all_findings(df1,df2,df3):
    for i,n in zip((df1,df2,df3),('Monocular','Binocular','LowCon')):
        print(n)
        print_anova_stats(i)
        print("\n\n")

if __name__=="__main__":
    plt.close('all')
    df1,df2,df3 = read_in_data()
    # print_all_findings(df1,df2,df3)
    # CollapsedModelPieChartAnovaFigure(df1,'left_only','eta').show()
    # CollapsedModelPieChartAnovaFigure(df2,'both_sides','eta').show()
    # CollapsedModelPieChartAnovaFigure(df3,'low_contrast','eta').show()
    # CollapsedModelCoefficientEstimatesFigure(df1).show()
    # CollapsedModelCoefficientEstimatesFigure(df2).show()
    # CollapsedModelCoefficientEstimatesFigure(df3).show()  
    plt.ioff()
    # fig = LickingModelFigure(df1)
    # fig.save("high_contrast_licking_pca")
    while True:
        try:
            LickingModelFigure(df1).save("unilat_highcon_licking")
            LickingModelFigure(df2).save("bilat_highcon_licking_pca")
            LickingModelFigure(df3).save("lowcon_licking_pca")
            break
        except ValueError:
            pass
        
    