# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:43:42 2020

@author: viviani
"""


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np



class PupilModelPredictionFigure:
    def __init__(self, 
                 path = "C:/Users/viviani/Desktop/pupilfactorpredictions.csv"):
        sb.set_style("dark")
        
        df = pd.read_csv(path)
        self.fig,ax = plt.subplots(ncols=2,nrows=2)
        #hits
        ax[0][0].set_title("Hits")
        self.plot_with_confidence_interval(ax[0][0], 
                                      df[df.trial_type=="hit"], 
                                      color="green",
                                      mean_color = "k")
        ax[0][0].legend()
        ax[0][0].set_ylabel("Pupil Diameter (pixels)")
        
        #misses
        ax[0][1].set_title("Misses")
        self.plot_with_confidence_interval(ax[0][1], 
                                      df[df.trial_type=="miss"], 
                                      color="darksalmon",
                                      mean_color = "k")
        
        #false alarms
        ax[1][0].set_title("False Alarms")
        self.plot_with_confidence_interval(ax[1][0], 
                                      df[df.trial_type=="fa"], 
                                      color="palegreen",
                                      mean_color = "k")
        ax[1][0].set_ylabel("Pupil Diameter (pixels)")
        ax[1][0].set_xlabel("Time since trial onset (s)")
        #correct rejections
        ax[1][1].set_title("Correct Rejections")
        self.plot_with_confidence_interval(ax[1][1], 
                                      df[df.trial_type=="cr"], 
                                      color="red",
                                      mean_color = "k")
        ax[1][1].set_xlabel("Time since trial onset (s)")
    
    def show(self):
        self.fig.show()
    
    @staticmethod
    def plot_with_confidence_interval(axis,dataframe, color, mean_color=None):
        if mean_color==None:
            mean_color=color
        x = np.arange(0,25)/5
        axis.plot(x,dataframe.fit[:25],
                  color = mean_color,
                  label = "prediction")
        axis.fill_between(x,dataframe.upr[:25],dataframe.lwr[:25],
                          color = color,
                          alpha = 0.3,
                          label = "80% confidence interval")
        axis.set_ylim((12,18))


class PupilModelPredictionFigureWithSingleAxis(PupilModelPredictionFigure):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.fig,ax = plt.subplots()
        self.plot_with_confidence_interval(ax, 
                                      df[df.trial_type=="hit"], 
                                      color="green")
        self.plot_with_confidence_interval(ax, 
                                      df[df.trial_type=="miss"], 
                                      color="darksalmon")
        self.plot_with_confidence_interval(ax, 
                                      df[df.trial_type=="fa"], 
                                      color="palegreen")
        self.plot_with_confidence_interval(ax, 
                                      df[df.trial_type=="cr"], 
                                      color="red")
        ax.set_ylabel("Pupil Diameter (pixels)")
        ax.set_xlabel("Time since trial onset (s)")


