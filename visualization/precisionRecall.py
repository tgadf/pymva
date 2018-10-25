#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:20:35 2018

@author: tgadfort
"""

from logger import info
from fsio import setFile

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
    
def plotPrecisionRecall(perfs, outdir, ext, pp = None):
    #info("Plotting Precision-Recall Curves for {0} Classifiers".format(len(perfs)))
    modelnames = perfs.keys()
    
    plt.figure()

    
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))    
    
    
    current_palette = sns.color_palette()
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i,modelname in enumerate(modelnames):
        perfdata  = perfs[modelname]
        precision = perfdata['PR']['precision']
        recall    = perfdata['PR']['recall']
        plt.plot(recall, precision,
                 label='{0}'.format(modelname),
                 color=current_palette[i], linestyle='-', linewidth=3)


    title = "Precision Recall Curve"
    value = title
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    
    
    
    if pp is not None:
        info("Saving {0} plot to multipage pdf".format(title), ind=4)
        pp.savefig()
    else:
        plotname = setFile(outdir, ".".join([value,ext]))
        info("Saving {0} plot to {1}".format(title, plotname), ind=4)
        plt.savefig(plotname)

    plt.close()
