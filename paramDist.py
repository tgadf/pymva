#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:43:21 2018

@author: tgadfort
"""

from logger import info
from targetInfo import isClassification, isRegression

from scipy.stats import randint as sp_randint
from scipy.stats._distn_infrastructure import rv_frozen

from scipy.stats import uniform as sp_randfloat


def getParamDist(config, modelname, nfeatures = None):
    info("Getting parameter distributions for {0}".format(modelname), ind=2)
    
    param_dist = None
    epsilon = 0.000001
    
    problemType = config['problem']
    
    treeParams = {"max_depth": [2, 4, 6, None],
                  "max_features": ['auto', 'sqrt', 'log2', None],
                  "min_impurity_decrease": sp_randfloat(0.0, 1-epsilon),
                  "min_samples_leaf": sp_randint(1, 10)}

    

    ###########################################################################
    ## rf, extratrees
    ###########################################################################
    if modelname in ["rf", "extratrees", "dtree", "gbm"]:
        param_dist = treeParams
        if modelname == "rf" or modelname == "extratrees":
            param_dist["bootstrap"] = [True, False]
        if modelname == "gbm":
            param_dist["learning_rate"] = sp_randfloat(0.01, 0.5)

        if modelname = ["rf", "extratrees", "dtree"]:
            if isClassification(problemType):
                param_dist["criterion"] = ["gini", "entropy"]
            if isRegression(problemType):
                param_dist["criterion"] = ["mae", "mse"]
        if isClassification(problemType):
            param_dist["criterion"] = ['mse', 'friedman_mse']
            param_dist["loss"] = ['deviance', 'exponential']
        if isRegression(problemType):
            param_dist["criterion"] = ["friedman_mse"]
            param_dist["loss"] = ['ls']
            
    ###########################################################################
    ## gbm
    ###########################################################################
    if modelname == "gbm":
        param_dist = {"max_depth": sp_randint(1, 10),
                      "max_features": ['auto', 'sqrt', 'log2', None],
                      "min_impurity_decrease": sp_randfloat(0.0, 1-epsilon),
                      "min_samples_leaf": sp_randint(1, 10)}
                        
            

    param_grid = {}
    for param,dist in param_dist.iteritems():
        param_grid[param] = getDistributionLimits(dist)
        

    if param_dist is None:
        info("Parameter distributions were not set for {0}".format(modelname), ind=4)
    else:
        info("Set grid and distributions for {0} parameters".format(len(param_dist)), ind=4)
        
    retval = {"dist": param_dist, "grid": param_grid}    
    return retval