# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:10:34 2020

@author: Vivian Imbriotis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11


class CollapsingFigureWithMockData:
    palette = sns.color_palette()
    @staticmethod
    def generate_mock_data():
        np.random.seed(432)
        X = np.linspace(0,5,25)
        tau = 1
        length = tau*8
        time = np.arange(0, length)
        kernel = np.exp(-time/tau)
        spike_prob = 0.1
        spike_size_mean = 0.8
        spike_times = 1.0 * (np.random.rand(25-kernel.size+1) < spike_prob)
        spike_amps = np.random.lognormal(mean = spike_size_mean, 
                                         sigma = 0.2, 
                                         size=spike_times.size) * spike_times
        spikes = 0.5*np.convolve(spike_amps, kernel)
        Y = np.random.normal(0.3,0.2,spikes.shape) + spikes
        return (X,Y)
    @staticmethod
    def collapse_across_time(data):
        point1 = data[:5].mean()
        point2 = data[5:15].mean()
        point3 = data[15:].mean()
        print(point1,point2,point3)
        return (point1, point2, point3)
    def __init__(self):
        fig,ax = plt.subplots(nrows=3, tight_layout=True,
                              figsize = [6,8])
        X,Y = self.generate_mock_data()
        ax[0].plot(X,Y, color = self.palette[0],label="Activity within a trial")
        ax[0].set_xlabel("Time (s)")
        ymin,ymax = ax[0].get_ylim()
        ax[0].vlines((0,1,3),ymin,ymax,linestyles='dashed',color = 'k')
        for name,pos in zip(('Tone','Stimulus','Response'),(0,1,3)):
            ax[0].text(pos+0.1,ymin,name,
                    horizontalalignment='left',
                    verticalalignment='bottom')
        ax[1].plot(X,Y, color=self.palette[0])
        collapsed_y = self.collapse_across_time(Y)
        collapsed_x = (0,1,3)
        bar_kwargs = {
                "height": collapsed_y,
                "x": collapsed_x,
                "width": (1,2,2),
                "alpha": 0.6,
                "align": 'edge',
                "linewidth": 0,
                "color": self.palette[1],
                "edgecolor": self.palette[1]
                }
        
        ax[1].bar(label = "Mean of response in each segment",**bar_kwargs)
        ax[1].set_xlabel("Time (s)")
        ax[2].set_ylim(ax[1].set_ylim())
        ax[2].plot((0.5,2,4),collapsed_y,color = 'k',marker='o',
                   label = 'Collapsed time series')
        ax[2].bar(**bar_kwargs)
        
        ax[2].set_xticks([0.5,2,4])
        ax[2].set_xticklabels(("Tone Bin", "Stimulus Bin", "Response Bin"))
        for a in ax:
            a.legend(loc="upper right")
            a.set_ylabel("Fluorescence ($\\Delta$F/F0 units)")
        fig.show()

if __name__=="__main__":
    CollapsingFigureWithMockData()