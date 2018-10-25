#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:43:59 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution, genPowerTen

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


###############################################################################
#
# LinearDiscriminantAnalysis
#
###############################################################################
def createLDAClassifier(params = None):
    info("Creating LDA Classifier", ind=4)
    
    ## Params
    params     = mergeParams(LinearDiscriminantAnalysis(), params)
    tuneParams = getLinearDiscriminantAnalysisParams()
    grid       = tuneParams['grid']

    
    info("With Parameters", ind=6)
    n_components = setParam('n_components', params, grid)
    info("Param: n_components = {0}".format(n_components), ind=6)
    
    solver = setParam('solver', params, grid)
    info("Param: solver = {0}".format(solver), ind=6)
    
    shrinkage = setParam('shrinkage', params, grid)
    info("Param: shrinkage = {0}".format(shrinkage), ind=6)
    
    
    ## Estimator
    clf = LinearDiscriminantAnalysis(n_components=n_components, 
                                     solver=solver, shrinkage=shrinkage)
    
    return {"estimator": clf, "params": tuneParams}


def getLinearDiscriminantAnalysisParams():
    params = {"n_components": [None],
              "solver": ['lsqr', 'eigen'],
              "shrinkage": ["auto"]}

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval



###############################################################################
#
# LinearDiscriminantAnalysis
#
###############################################################################
def createQDAClassifier(params = None):
    info("Creating QDA Classifier", ind=4)
    
    ## Params
    params     = mergeParams(QuadraticDiscriminantAnalysis(), params)
    tuneParams = getQuadraticDiscriminantAnalysisParams()
    grid       = tuneParams['grid']

    
    info("With Parameters", ind=6)
    reg_param = setParam('reg_param', params, grid)
    info("Param: reg_param = {0}".format(reg_param), ind=6)
    
    
    ## Estimator
    clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    
    return {"estimator": clf, "params": tuneParams}


def getQuadraticDiscriminantAnalysisParams():
    params = {"reg_param": genPowerTen(-4, 4, 100)}

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval