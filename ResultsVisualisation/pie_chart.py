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
from matplotlib.ticker import PercentFormatter
from scipy.stats import combine_pvalues
import json
sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

readable_titles = {
    "anova_segment_pvalue" : "Effect of Trial Segment",
    "anova_correct_pvalue":  "Effect of Correct Response",
    "anova_go_pvalue":       "Effect of Go vs No-Go",
    "anova_side_pvalue" :    "Effect of Stimulus Side",
    "anova_contrast_pvalue": "Effect of Stimulus Contrast",
    "anova_segment_fvalue" : "Effect of Trial Segment",
    "anova_correct_fvalue" : "Effect of Correct Response",
    "anova_go_fvalue" :      "Effect of Go vs No-Go",
    "anova_side_fvalue" :    "Effect of Stimulus Side",
    "anova_contrast_fvalue": "Effect of Stimulus Contrast",
    "anova_segment_eta" :    "Effect of Trial Segment",
    "anova_correct_eta" :    "Effect of Correct Response",
    "anova_go_eta" :         "Effect of Go vs No-Go",
    "anova_side_eta" :       "Effect of Stimulus Side",
    "anova_contrast_eta":    "Effect of Stimulus Contrast"
    }

def count_unique_index(df, by):                                                                                                                                                 
    return df.groupby(by).size().reset_index().rename(columns={0:'count'})

class MeanChangeInFluorescenceDuringTrials:
    def __init__(self,array):
        self.fig, ax = plt.subplots(figsize=[8,4],tight_layout=True)
        ax.set_xlabel("Mean change in fluorescence during trials ($\Delta$F/F0 units)")
        ax.set_ylabel("Proprotion of ROIs")
        ax.hist(array,weights=np.ones(len(array))/len(array),
                bins = 30)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    def show(self):
        self.fig.show()
        

class CollapsedModelPieChartAnovaFigure:
    colors = sns.color_palette()
    reserved_colors = colors[:5]
    unreserved_colors = colors[5:]
    color_mapping = {
        'Trials':  reserved_colors[0],
        'Trials+Go':      reserved_colors[1],
        'Trials+Correct': reserved_colors[2],
        'Trials+Side':    reserved_colors[3],
        'Trials+Contrast':reserved_colors[4],
        'anova_segment_eta':reserved_colors[0],
        'anova_go_eta':reserved_colors[1],
        'anova_correct_eta':reserved_colors[2],
        'anova_side_eta':reserved_colors[3],
        'anova_contrast_eta':reserved_colors[4],
        'None': 'gray',
        'Others': (0.7,0.7,0.7)
        }
    def get_colors(self,counter_dataframe):
        results = []
        for idx,row in counter_dataframe.iterrows():
            name = row["name"]
            results.append(self.get_color(name,idx))
        return results
    
    def get_color(self,name,idx):
        if name in self.color_mapping:
            return self.color_mapping[name]
        else:
            return self.unreserved_colors[idx%len(self.unreserved_colors)]

        
    def __init__(self,df,dataset='left_only',statistic='f'):
        pvalue_cols = [c for c in df.columns if 'pvalue' in c and 'anova' in c]
        fvalue_cols = [c for c in df.columns if 'fvalue' in c and 'anova' in c]
        eta_cols    = [c for c in df.columns if 'eta'    in c and 'anova' in c]
        
        pvals = df[pvalue_cols]
        fvals = df[fvalue_cols]
        evals = df[eta_cols]
        if statistic.lower() in ('f','f value','fvalue'):
            mode = 'f'
        elif statistic.lower() in ('e','eta','eta2','eta squared'):
            mode = 'eta2'
        else:
            raise ValueError(f"statistic muse be 'f' or 'eta2', not {statistic}")
        if dataset not in ('left_only','both_sides','low_contrast'):
            raise ValueError(
                f"Dataset must be one of ('left_only','both_sides',"+
                f"'low_contrast'), not {dataset}")
    
        counter = count_unique_index((pvals<0.05),pvalue_cols)
        counter['percent']=counter['count']/counter['count'].sum()
        if dataset=='left_only':     names = np.array(('Trials','Correct','Go'))
        elif dataset=='both_sides':  names = np.array(('Trials','Correct','Go',
                                                    'Side'))
        elif dataset=='low_contrast':names  = np.array(('Trials','Correct','Go',
                                                    'Side','Contrast'))
        
        names = [names[boolrow[:-2].to_numpy().astype('bool')] for _,boolrow in counter.iterrows()]
        names = np.array(['+'.join(ls) if list(ls) else "None" for ls in names])
        counter['name'] = names
        
        #Begin by setting up the figure, with one set of axes on the left
        #and a column of axes on the right
        self.fig = plt.figure(figsize = [12,8], tight_layout=True)
        gs = GridSpec(len(fvals.columns),2,figure=self.fig)
        right_ax = []
        left_ax = self.fig.add_subplot(gs[:,0])
        
        #Make histograms of effect sizes on each axis on the right.
        #Use either f-value or partial eta-squared as a measure of effect size
        if mode=='f':
            for idx,c in enumerate(fvals.columns):
                print(c)
                right_ax.append(
                    self.fig.add_subplot(gs[idx,1])
                    )
                right_ax[idx].hist(fvals[c][pvals[c.replace('fvalue','pvalue')]<0.05],
                             label = readable_titles[c],
                             color = self.get_color(name,idx),
                             bins = 50 if idx==0 else 4)
                right_ax[idx].legend()
                right_ax[idx].set_xlim(0,0.175)
                ymax = max([a.get_ylim()[1] for a in right_ax])
                right_ax[idx].set_ylim(None,ymax)
                           
        elif mode=='eta2':
            for idx,c in enumerate(evals.columns):
                right_ax.append(
                    self.fig.add_subplot(gs[idx,1])
                    )
                right_ax[idx].hist(evals[c][pvals[c.replace('eta','pvalue')]<0.05],
                             weights = np.ones(
                                 (pvals[c.replace('eta','pvalue')]<0.05).sum())/len(evals),
                             label = readable_titles[c],
                             color = self.get_color(c,idx),
                             bins = np.linspace(0,0.175,50))
                right_ax[idx].legend()
                right_ax[idx].set_xlim(0,0.175)
                ymax = max([a.get_ylim()[1] for a in right_ax])
                right_ax[idx].set_ylim(None,ymax)
                right_ax[idx].yaxis.set_major_formatter(PercentFormatter(1))
                
        #Add axis labels and clean up any excess text
        right_ax[len(right_ax)//2].set_ylabel("Proportion of models")
        right_ax[-1].set_xlabel("F value" if mode=='f' else "Partial $\eta^2$")
        for a in right_ax[:-1]: a.set_xticklabels([])
        
        
        #Now to build the pie chart
        #We have a lot of possible combinations of significant variables, so
        #we need to group all the ones that were only significant in a small
        #number of rois into a category called 'other'
        modified_counter = counter[['name','count','percent']][counter.percent>0.01]
        others_row = ('Others',counter['count'][counter.percent<=0.01].sum(),
                      counter.percent[counter.percent<=0.01].sum())
        modified_counter = modified_counter.append(
            pd.Series(others_row,index=(['name','count','percent'])),
            ignore_index=True)
        
        #Now create the wedges of the pie chart
        wedges,text1,text2 = left_ax.pie(modified_counter['count'],
                                autopct='%1.f%%',counterclock=True,
                                startangle=60, pctdistance=0.9,
                                colors = self.get_colors(modified_counter))
        
        #Each wedge needs a label with a line connecting the wedge to the
        #label
        kw = dict(arrowprops=dict(arrowstyle="-",color='black'),
                  zorder=5, va="center")
        
        annotations = []
        for wedge,(idx,variable) in zip(wedges,modified_counter.iterrows()):
            ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            text_xy = (x*1.2,y*1.2)
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            annotations.append(
                left_ax.annotate(variable['name'], xy=(x, y), 
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

class CollapsedModelPValuesFigure:
    colors = sns.color_palette()
    reserved_colors = colors[:5]
    unreserved_colors = colors[5:]
    color_mapping = {
        'anova_segment_p_unadjusted':reserved_colors[0],
        'anova_go_p_unadjusted':reserved_colors[1],
        'anova_correct_p_unadjusted':reserved_colors[2],
        'anova_side_p_unadjusted':reserved_colors[3],
        'anova_contrast_p_unadjusted':reserved_colors[4],
        }
    def __init__(self,df):
        anova_pvals = df[[c for c in df.columns if 'anova' in c and 'p_unadjusted' in c]]
        self.fig,ax = plt.subplots(ncols = len(anova_pvals.columns),
                              figsize = [12,4],
                              tight_layout=True)
        for idx,c in enumerate(anova_pvals.columns):
            ax[idx].set_title(readable_titles[c.replace("p_unadjusted","pvalue")])
            ax[idx].hist(anova_pvals[c].values,
                         weights = np.ones(len(anova_pvals[c]))/len(anova_pvals[c]),
                         bins=np.linspace(0,1,50//(len(anova_pvals.columns)-1),
                                          endpoint=True),
                         color = self.color_mapping[c])
            ax[idx].set_xlabel("p-value")
            ax[idx].set_ylabel("Percent of models")
            ax[idx].yaxis.set_major_formatter(PercentFormatter(1))
    def show(self):
        self.fig.show()
            
class CollapsedModelCoefficientEstimatesFigure:
    def __init__(self,df):
        colnames = ("Response Bin","Stimulus Bin","Tone Bin")
        coefs = df[[c for c in df.columns if ('coefficient' in c and 
                                                'estimate' in c and
                                                'lick' not in c)]]
        intercept = coefs.coefficient_x_intercept_estimate.to_numpy()
        intercept = np.stack([intercept]*3).transpose()
        main_effect = coefs.iloc[:,0:3]
        main_effect.columns = colnames
        main_effect.loc[:,"Response Bin"] = 0
        correct = coefs[[c for c in coefs.columns if 'correct1' in c and 'contrast' not in c]]
        go      =  coefs[[c for c in coefs.columns if 'go1' in c]]
        print(correct.columns)
        correct.columns = colnames
        go.columns      = colnames
        main_effect = main_effect[reversed(colnames)]
        correct = correct[reversed(colnames)]
        go = go[reversed(colnames)]
        self.fig, ax = plt.subplots(ncols = 2, nrows = 2,
                                    tight_layout=True,
                                    figsize = [12,8])
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
                                                'estimate' in c and
                                                'lick' in c)]]
        
        intercept = coefs.lick_coefficient_x_intercept_estimate.to_numpy()
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
        pca = PCA(n_components = 3)
        kernels_in_pcs = pca.fit_transform(kernels.to_numpy())
        pc1,pc2,pc3 = pca.components_
        self.fig3, pca_ax = plt.subplots(ncols=3, tight_layout=True,
                                         figsize = [8,3])
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


    
def read_in_data():
    file = open("pythonic_column_names.json",'r')
    d = json.load(file); file.close()
    df1 = pd.read_csv("../RScripts/results_left_only.csv").rename(
        d,axis='columns')
    df2 = pd.read_csv("../RScripts/results_binocular.csv").rename(
        d,axis='columns')
    df3 = pd.read_csv("../RScripts/results_low_contrast.csv").rename(
        d,axis='columns')
    df4 = pd.read_csv("../RScripts/results_binocular_fullkernel.csv")
    
    return(df1,df2,df3,df4)


class SubtypedROIsWithSignificantTrialResponseFigure:
    def __init__(self,df):
            pvals_on_anova = df.anova_segment_pvalue
            bins_coefs = df[['coefficient_stimulus_estimate',
                             'coefficient_tone_estimate']]
            bins_coefs.columns = ["Stimulus Bin","Tone Bin"]
            bins_coefs['Response Bin'] = 0
            cols = bins_coefs.columns
            results_max = bins_coefs.idxmax(axis=1).value_counts()[cols]
            results_min = bins_coefs.idxmin(axis=1).value_counts()[cols]
            self.fig,ax = plt.subplots(ncols = 2,tight_layout=True,
                                       figsize = [8,4])
            ax[0].set_title("ROIs by time of most activity")
            ax[0].pie(results_max.values,labels = results_max.index,
                      autopct='%1.1f%%')
            ax[1].set_title("ROIs by time of least activity")
            ax[1].pie(results_min.values,labels=results_min.index,
                      autopct='%1.1f%%')
    def show(self):
        self.fig.show()
            
        
def print_all_findings(df1,df2,df3):
    for i,n in zip((df1,df2,df3),('Monocular','Binocular','LowCon')):
        print(n)
        print_anova_stats(i)
        print("\n\n")

def print_anova_stats(df):
    anova_pvals = df[[c for c in df.columns if 'anova' in c and 'pvalue' in c]]
    anova_unadjusted = df[[c for c in df.columns if 
                           'anova' in c and 'p_unadjusted' in c]]
    
    print("NUMBER OF SIGNIFICANT ROIS")
    print(f"total = {len(df)}")
    print((anova_pvals<0.05).sum())
    unpredicted = ((anova_pvals<0.05).sum(axis='columns')==0).sum()
    print(f"with no significant predictors    {unpredicted}")
    print("\nPERCENTAGE SIGNIFICANT ROIS")
    print(100*(anova_pvals<0.05).sum()/len(anova_pvals))
    print(f"with no significant predictors    {100*unpredicted/len(anova_pvals)}")
    for var in anova_unadjusted.columns:
        stat, p = combine_pvalues(anova_unadjusted[var])
        print(f"{var} was {'NOT' if p>(0.05/12) else ''} significant")
        print(f"        (Fisher's Combined Test, chi2={stat:.4f}, p={p:.4f})")
        

class TrialKernelFigure:
    x_for_trials = np.linspace(0,5,25)
    x_for_licks = np.linspace(-2,2,21)
    def __init__(self, df, dataset = 'bilateral'):
        self.dataset = dataset
        coefs = [c for c in df.columns if 'coefficient' in c and 'estimate' in c]
        self.lick  = [c for c in coefs if 'lick_factor' in c]
        self.trial = [c for c in coefs if all(('trial_factor' in c,
                                         'correct' not in c,
                                         'go' not in c,
                                         'contrast' not in c,
                                         'side' not in c))]
        self.correct = [c for c in coefs if 'correct' in c]
        self.go      = [c for c in coefs if 'go' in c]
        self.side    = [c for c in coefs if 'side' in c]
        self.rois = df[df['overall.model.adj.rsquared']>0.3]
        self.figures = []
        for idx,roi in self.rois.iterrows():
            self.figures.append(self.create_figure(roi))
    def create_figure(self,roi):
        intercept   = roi['coefficient X.Intercept. estimate']
        trial       = np.append(np.zeros(1),roi[self.trial].to_numpy())
        lick        = roi[self.lick].to_numpy()
        correct     = roi[self.correct].to_numpy()
        go          = roi[self.go].to_numpy()
        side        = roi[self.side].to_numpy()
        typical = intercept + trial + (correct + go + side)/2
        hits    = intercept + trial + correct + go + (side/2)
        misses  = intercept + trial + go + (side)/2
        fas     = intercept + trial + (side)/2
        crs     = intercept + trial + correct + (side)/2
        fig,ax = plt.subplots(nrows = 3, ncols = 2,
                              figsize = [8,9],
                              tight_layout=True)

        ax[0][0].set_title("Hit Trials")
        ax[0][0].plot(self.x_for_trials, hits)
        ax[0][1].set_title("Miss Trials")
        ax[0][1].plot(self.x_for_trials, misses)
        ax[1][0].set_title("False Alarm Trials")
        ax[1][0].plot(self.x_for_trials, fas)
        ax[1][1].set_title("Correct Rejection Trials")
        ax[1][1].plot(self.x_for_trials, crs)
        ax[2][0].set_title("Average Trial")
        ax[2][0].plot(self.x_for_trials,typical)
        
        for a in ax.flatten()[:-1]:
            a.set_xlabel("Time since trial onset")
            a.set_ylim(-0.5,1)
            a.set_xlabel('Time within trial (s)')
            ymin,ymax = a.get_ylim()
            a.vlines((0,1,3),ymin,ymax,linestyles='dashed',color = 'k')
            for name,pos in zip(('Tone','Stimulus','Response'),(0,1,3)):
                a.text(pos+0.1,ymin,name,
                        horizontalalignment='left',
                        verticalalignment='bottom')
        for a in ax[:,0]:
            a.set_ylabel("Fluorescence ($\Delta$F/F0 units)")
        ax[2][1].set_title("Licking")
        ax[2][1].set_ylim(-0.5,1)
        ax[2][1].plot(self.x_for_licks,lick)
        ax[2][1].vlines(0,-0.5,1,linestyles='dashed',color='k')
        ax[2][1].text(0.1,-0.2,'Lick event',ha='left',va='bottom')
        ax[2][1].set_xlabel("$\Delta$t around licks")
        
        return fig
    def show(self):
        for figure in self.figures: figure.show()
                  



if __name__=="__main__":
    plt.close('all')
    df1,df2,df3,df4 = read_in_data()
    # print_all_findings(df1,df2,df3)
    plt.ioff()
    MeanChangeInFluorescenceDuringTrials(
        pd.read_csv("../RScripts/during_trials.csv").model_estimates.values
        ).show()
    # CollapsedModelPValuesFigure(df1).show()
    # CollapsedModelPValuesFigure(df2).show()
    # CollapsedModelPValuesFigure(df3).show()
    # CollapsedModelPieChartAnovaFigure(df1,'left_only','eta').show()
    # CollapsedModelPieChartAnovaFigure(df2,'both_sides','eta').show()
    # CollapsedModelPieChartAnovaFigure(df3,'low_contrast','eta').show()
    # CollapsedModelCoefficientEstimatesFigure(df1).show()
    # CollapsedModelCoefficientEstimatesFigure(df2).show()
    # CollapsedModelCoefficientEstimatesFigure(df3).show()  
    # SubtypedROIsWithSignificantTrialResponseFigure(df1).show()
    # SubtypedROIsWithSignificantTrialResponseFigure(df2).show()
    # SubtypedROIsWithSignificantTrialResponseFigure(df3).show()
    # TrialKernelFigure(df4).show()
    # LickingModelFigure(df1).show()
    # plt.ioff()
    # fig = LickingModelFigure(df1)
    # fig.save("high_contrast_licking_pca")
    # while True:
    #     try:
    #         LickingModelFigure(df1).save("unilat_highcon_licking")
    #         LickingModelFigure(df2).save("bilat_highcon_licking_pca")
    #         LickingModelFigure(df3).save("lowcon_licking_pca")
    #         break
    #     except ValueError:
    #         pass


