#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:34:19 2018

@author: tgadfort
"""

from logger import info
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from targetInfo import getTargetNames

from fsio import setFile
from numpy import newaxis, arange


def plotConfusionMatrix(perfs, config, outdir, ext, pp = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title='Confusion Matrix'
    normalize=True
    cmap=plt.cm.Blues
    
    try:
        cm = perfs['xgboost']['Confusion']['matrix']
    except:
        return
    classes = getTargetNames(config)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
    
    value = title
    plotname = setFile(outdir, ".".join([value,ext]))
    info("Saving {0} plot to {1}".format(title, plotname), ind=4)
    plt.savefig(plotname)
    
    plt.close()