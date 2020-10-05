# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:57:25 2020

@author: Vivian Imbriotis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
sns.set_style('dark')
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

class BothStimuliFigure:
    x = np.linspace(0,10,100)
    y = np.linspace(0,10,100)
    X,Y = np.meshgrid(x,y)
    def __init__(self):
        Z_go = self.X%2 > 1
        Z_no_go = self.Y%2 > 1
        self.fig,ax = plt.subplots(ncols=2, tight_layout=True)
        self.image1 = ax[0].imshow(Z_go,cmap='gray')
        self.image2 = ax[1].imshow(Z_no_go,cmap='gray')
        ax[0].set_title("Go Stimulus")
        ax[1].set_title("No Go Stimulus")
        for a in ax: a.set_xticks([]); a.set_yticks([])
        self.anim = FuncAnimation(self.fig,self.update_image,20,interval=50)
    def go_stim(self,t):
        Z_go = (self.X-t*0.1)%2 > 1; return Z_go
    def nogo_stim(self,t):
        Z_nogo = (self.Y+t*0.1)%2 > 1; return Z_nogo
    
    def update_image(self,t):
        self.image1.set_data(self.go_stim(t))
        self.image2.set_data(self.nogo_stim(t))
    def show(self):
        self.fig.show()
    def save(self,name):
        self.anim.save(f"{name}.gif",PillowWriter(fps=20))

class SingleMiniStimulusThumbnailForFlowchart:
    x = np.linspace(0,10,100)
    y = np.linspace(0,10,100)
    X,Y = np.meshgrid(x,y)
    def __init__(self, go = True):
        self.go = go
        Z_go = self.X%2 >= 1
        Z_no_go = self.Y%2 >= 1
        self.fig,ax = plt.subplots(tight_layout=True,figsize = [2,2])
        self.image = ax.imshow(Z_go if go else Z_no_go,cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        self.anim = FuncAnimation(self.fig,self.update_image,20,interval=100)
    def go_stim(self,t):
        Z_go = (self.X-t*0.1)%2 > 1; return Z_go
    def nogo_stim(self,t):
        Z_nogo = (self.Y+t*0.1)%2 > 1; return Z_nogo
    
    def update_image(self,t):
        if self.go:
            self.image.set_data(self.go_stim(t))
        else:
            self.image.set_data(self.nogo_stim(t))
    def show(self):
        self.fig.show()
    def save(self,name):
        self.anim.save(f"{name}.gif",PillowWriter(fps=10))

if __name__=="__main__":
    SingleMiniStimulusThumbnailForFlowchart().save('go')
    SingleMiniStimulusThumbnailForFlowchart(go=False).save('nogo')
