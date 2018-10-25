#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:22:16 2017

@author: tgadfort
"""

import sys
if '/Users/tgadfort/Python' not in sys.path:
    sys.path.insert(0, '/Users/tgadfort/Python')
if '/Users/tgadfort/Documents/pymva' not in sys.path:
    sys.path.insert(1, '/Users/tgadfort/Documents/pymva')

from fsio import setFile, isFile
from fileio import get


from logger import info, setupLogger
from loadData import readData
from formatData import formatData
from splitData import getTrainTestData, isSplitDataReady, loadTrainTestData, getTruthData
from trainModel import trainModel, testModel, getModelFileName, getTrainedModel, saveTrainedModel
from performance.summary import getModelPerformance
from visualization.summary import plotResults
from targetInfo import isClassification, isRegression

### Functions needed for interactive use
#import regression, classification
from models import getModels, getModel
from splitData import getTrainData


setupLogger()
def loadConfig():
    configname = setFile("/Users/tgadfort/Documents/pymva", "config.yaml")
    info("Importing [{0}]".format(configname), ind=0)
    config     = get(configname)
    return config


def createData(config):
    if isSplitDataReady(config):
        ### Load split data
        X_train, X_test, X_valid, y_train, y_test, y_valid = loadTrainTestData(config)
    else:
        ### Load data based on config
        info("Loading data", ind=0)
        pddf = readData(config)
    
    
        ### Format data
        info("Formatting data", ind=0)
        pddf = formatData(pddf, config)
    
    
        ### Split data
        info("Splitting data", ind=0)
        X_train, X_test, X_valid, y_train, y_test, y_valid = getTrainTestData(pddf, config)


    return X_train, X_test, X_valid, y_train, y_test, y_valid


def runModels(config, models = None, force = False):

    X_train, X_test, X_valid, y_train, y_test, y_valid = createData(config)
    
    if not isinstance(models, list):
        models = [models]
    
    perfs = {}
    for modelname in models:
        modelFileName = getModelFileName(config, modelname)
        if isFile(modelFileName):
            info("Already have {0} estimator.".format(modelname))
            if force is False:
                continue
            else:
                info("Will rerun {0} estimator.".format(modelname))
        
        clf     = trainModel(modelname, X_train, y_train, config)
        tval    = testModel(modelname, clf, X_test, config)
        perf    = getModelPerformance(y_test, tval, config)
        
        modelResults = {"name": modelname, "estimator": clf, "test": tval, "perf": perf}
        info("Saving {0} estimator".format(modelname), ind=2)
        saveTrainedModel(config, modelResults)
        perfs[modelname] = perf
        
        
    
def getPerformanceData(config, models):
    perfs  = {}
    for modelname in models:
        info("Loading {0} estimator".format(modelname), ind=2)
        modelResults = getTrainedModel(config, modelname)
        perfs[modelname] = modelResults['perf']
        
    return perfs

    
def plotPerformance(config, models):
    info("Getting Truth Data", ind=2)
    y_test = getTruthData(config)
    perfs = getPerformanceData(config, models)            
    plotResults(perfs, y_test, config)