# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:41:51 2020

@author: Vivian Imbriotis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

sns.set_style("dark")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

class LossDueToResidualFigure:
    x = np.linspace(-3,3,200)
    def cost(self,residuals, k=10,delta=1):
        upper_cost = (residuals>0)*(delta**2 * (np.sqrt(1 + (residuals/delta)**2)-1))
        lower_cost = (residuals < 0)*(k)*(residuals**(2))
        return upper_cost + lower_cost
    def update(self,frame):
        k = (2*frame)
        if k==0: k=1
        y = self.cost(self.x,k=k)
        self.klabel.set_text("$\\bf{" + f"k = {k}" + "}$")
        self.line.set_data(self.x,y)
    def __init__(self):
        self.fig,ax = plt.subplots()
        ax.set_xlim((-3,3))
        ax.set_ylim((-0.5,5))
        ax.set_ylabel("Loss due to residual")
        ax.set_xlabel("Residual")
        self.line, = ax.plot([],label = 'Bespoke loss function',
                             color = 'k')
        ax.plot(self.x,self.x**2, label = "Least-squares loss",
                linestyle = "--")
        ax.plot(self.x,np.abs(self.x),label = "L1 norm loss",
                linestyle = "--")
        ax.legend()
        
        bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        self.klabel = ax.text(x=2.8,y=4.3,ha='right',va='bottom',
                              s = "", bbox = bbox_props)
        
        self.anim = FuncAnimation(self.fig,self.update,frames=20,
                                  repeat=True,repeat_delay=0)
    def show(self):
        self.fig.show()
    def save(self,name):
        self.anim.save(f'{name}.gif', writer=PillowWriter(fps=5))
        
if __name__=="__main__":
    plt.close('all')
    fig = LossDueToResidualFigure().save("val")
    