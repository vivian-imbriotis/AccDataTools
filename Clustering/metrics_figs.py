# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:52:41 2020

@author: uic
"""

import matplotlib.pyplot as plt
from string import ascii_lowercase


def edit_map(str1,str2):
    res = []
    for a,b in zip(str1,str2):
        res.append(a!=b)
    return res

edit_dist = lambda str1,str2:sum(edit_map(str1,str2))


'absolutely'
'conclusion'

a='retired'
b='refined'



def format_table(str1,str2):
    fig, axes = plt.subplots(2,1)
    
    data = [[],[],[]]
    n = min(len(str1),len(str2))
    data[0] = list(str1)[0:n-1]
    data[1] = list(str2)[0:n-1]
    data[2] = list(map(int,edit_map(data[0],data[1])))
    for row in data:
        print(row)
    axes[0].axis('tight')
    axes[0].axis('off')
    axes[0].table(
        cellText = data,
#        colWidths = [2/len(data[0])] + [1/len(data[0])]*len(data[0]-1),
        rowLabels = ['First String',
                 'Second String',
                 'Edit Required?'],
        loc = 'center'
        )
    return fig

fig = format_table(a,b)
fig.show()