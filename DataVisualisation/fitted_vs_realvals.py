# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:26:34 2020

@author: Vivian Imbriotis
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn
seaborn.set()

def main():
    data = pd.read_csv("C:\\Users\\Vivian Imbriotis\\Documents\\vivFittedVsReal.csv")
    fig,ax = plt.subplots()
    real_data = ax.plot(data["roi.dF_on_F"], color = 'black')[0]
    print(real_data)
    real_data.set_label("Real data")
    fitted_vals = ax.plot(data["model.fitted.values"], color='red')[0]
    fitted_vals.set_label("Fitted linear model based on licking behavior")
    ax.legend()
    ax.set_title("ROI 2016-05-31_02_CFEB013 [169]")
    fig.show()
    return data

if __name__=="__main__":
    data=main()