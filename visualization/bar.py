#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:37:39 2018

@author: tgadfort
"""

from logger import info
import matplotlib.pyplot as plt
import seaborn as sns

from fsio import setFile


def plotBar(perfs, value, title, outdir, ext, pp = None):
    sns.set_style("whitegrid")

    modelnames = perfs.keys()

    values = [perfs[x][value] for x in modelnames]
    ax = sns.barplot(x=modelnames, y=values)
    ax.set_title(title)
 
    for item in ax.get_xticklabels():
        item.set_rotation(45)
    #plt.show()
    
    if pp is not None:
        info("Saving {0} plot to multipage pdf".format(title), ind=4)
        pp.savefig()
    else:
        plotname = setFile(outdir, ".".join([value,ext]))
        info("Saving {0} plot to {1}".format(title, plotname), ind=4)
        plt.savefig(plotname)

    plt.close()
    
    
def plotKappa(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'Kappa', 'Cohen Kappa', outdir, ext, pp)
    
def plotAccuracy(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'Accuracy', 'Accuracy', outdir, ext, pp)

def plotPrecision(perfs, outdir, ext, pp = None):    
    plotBar(perfs, 'AveragePrecision', 'Average Precision', outdir, ext, pp)

def plotRecall(perfs, outdir, ext, pp = None):    
    plotBar(perfs, 'Recall', 'Average Recall', outdir, ext, pp)

def plotLogLoss(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'LogLoss', 'Log Loss', outdir, ext, pp)

def plotMAE(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'MeanAbsoluteError', 'Mean Absolute Error', outdir, ext, pp)

def plotMSE(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'MeanSquaredError', 'Mean Squared Error', outdir, ext, pp)

def plotR2(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'R2Score', 'R2 Score', outdir, ext, pp)

def plotExplainedVariance(perfs, outdir, ext, pp = None):
    plotBar(perfs, 'ExplainedVariance', 'Explained Variance', outdir, ext, pp)
    
    