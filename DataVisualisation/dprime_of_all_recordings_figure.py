# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle

results = pickle.load(open('C:\\Users\\uic\\Desktop\\resultsdump.p','rb'))


cols = int(len(results)**0.5)
rows = len(results)//cols + 1

figure, axes = plt.subplots(rows,cols, figsize = (12,12))
axes = axes.flatten()

for idx, (mouse, (experiment,dprimes)) in enumerate(results.items()):
    dprimes.sort()
    xords = np.arange(len(dprimes))
    axes[idx].bar(xords, dprimes, 0.8, align='edge')
    axes[idx].set_title(mouse[mouse.rfind('\\')+1:])
    axes[idx].set_xlabel(f'Trials (n={len(dprimes)})')
    axes[idx].set_ylabel('Dprime')
    axes[idx].set(xlim=(0,len(dprimes)), ylim=(-6,6))
    axes[idx].set_xbound(lower=0, upper=len(dprimes))

for idx, axis in enumerate(axes):
    if idx>=len(results):
        axis.remove()

plt.tight_layout()
plt.savefig('figure.png')