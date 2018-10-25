#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:18:14 2018

@author: tgadfort
"""

import numpy as np
from logger import info
import matplotlib.pyplot as plt
import seaborn as sns

from fsio import setFile

def plotResidualsAndPrediction(perfs, y_test, outdir, ext, pp = None):
    sns.set(style="whitegrid")
    
    modelnames = perfs.keys()
    x = y_test
    x.name = "Truth"

    for i,modelname in enumerate(modelnames):
        y = perfs[modelname]['Residuals']
        title = "{0} Residuals And Prediction".format(modelname)
        y.name = "Residuals"
    
        # Plot the residuals after fitting a linear model
        ax = sns.residplot(x, y, lowess=True, color="b")
        ax.set_title(title)
        
        value = title
        plotname = setFile(outdir, ".".join([value,ext]))
        info("Saving {0} plot to {1}".format(title, plotname), ind=4)
        plt.savefig(plotname)

    
    
def plotResiduals(perfs, outdir, ext, pp = None):
    sns.set(style="whitegrid")
    
    modelnames = perfs.keys()

    for i,modelname in enumerate(modelnames):
        y    = perfs[modelname]['Residuals']
        miny = np.percentile(y, 1)
        maxy = np.percentile(y, 99)
        capy = np.copy(y)
        capy[capy < miny] = miny
        capy[capy > maxy] = maxy
        y.name = "Residuals"
        ax = sns.distplot(capy, rug=False, label=modelname)
        
    title = "Residuals"
    ax.set_title(title)
    ax.legend()

    value = "Residuals"
    plotname = setFile(outdir, ".".join([value,ext]))
    info("Saving {0} plot to {1}".format(title, plotname), ind=4)
    plt.savefig(plotname)
    plt.close()
