#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:31:58 2018

@author: tgadfort
"""

from logger import info
from targetInfo import isClassification, isRegression

from mvapath import getPlotsDir
from bar import plotKappa, plotPrecision, plotRecall, plotLogLoss, plotAccuracy
from bar import plotMAE, plotMSE, plotR2, plotExplainedVariance
from roc import plotROC
from res import plotResiduals
from precisionRecall import plotPrecisionRecall
from confusionMatrix import plotConfusionMatrix

from fsio import setFile
from matplotlib.backends.backend_pdf import PdfPages

def plotResults(perfs, y_truth, config):
    info("Making Performance Plots", ind=0)

    outdir = getPlotsDir(config)
    performanceConfig = config['performance']
    ext         = performanceConfig['ext']
    isPdf       = ext == 'pdf'
    isMultipage = performanceConfig['multipage']
    if isMultipage and isPdf:
        pdfname = setFile(outdir, 'results.pdf')
        info("Saving all performance plots to {0}".format(pdfname), ind=2)
        pp = PdfPages(pdfname)
    else:
        info("Saving all performance plots individually as {0}".format(ext), ind=2)
        pp = None

    
    badModels = [x for x in perfs.keys() if len(perfs[x]) == 0]
    for modelname in badModels:
        info("Not plotting {0}".format(modelname))
        del perfs[modelname]
    
    
    if isClassification(config['problem']):
        plotKappa(perfs, outdir, ext, pp)
        plotPrecision(perfs, outdir, ext, pp)
        plotRecall(perfs, outdir, ext, pp)
        plotLogLoss(perfs, outdir, ext, pp)
        plotAccuracy(perfs, outdir, ext, pp)
        plotPrecisionRecall(perfs, outdir, ext, pp)
        plotROC(perfs, outdir, ext, pp)
        plotConfusionMatrix(perfs, config, outdir, ext, pp)
    
    if isRegression(config['problem']):
        plotMAE(perfs, outdir, ext, pp)
        plotMSE(perfs, outdir, ext, pp)
        plotExplainedVariance(perfs, outdir, ext, pp)
        plotR2(perfs, outdir, ext, pp)
        plotResiduals(perfs, outdir, ext, pp)

    if isMultipage and isPdf:
        info("Closing multipage pdf", ind=2)
        pp.savefig()
        pp.close()
    