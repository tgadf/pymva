#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:43:14 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution

from sklearn.tree import DecisionTreeClassifier

from scipy.stats import randint as rint
from scipy.stats import uniform as rfloat

###############################################################################
#
# Decision Tree Classifier
#
###############################################################################
def createDecisionTreeClassifier(params):
    info("Creating Decision Tree Classifier", ind=4)
    
    ## Params
    params     = mergeParams(DecisionTreeClassifier(), params)
    tuneParams = getDecisionTreeClassifierParams()
    grid       = tuneParams['grid']
    
    info("With Parameters", ind=6)
    criterion = setParam('criterion', params, grid)
    info("Param: criterion = {0}".format(criterion), ind=6)
    
    max_depth = setParam('max_depth', params, grid)
    info("Param: max_depth = {0}".format(max_depth), ind=6)
    
    max_features = setParam('max_features', params, grid)
    info("Param: max_features = {0}".format(max_features), ind=6)
    
    max_leaf_nodes = setParam('max_leaf_nodes', params, grid)
    info("Param: max_leaf_nodes = {0}".format(max_leaf_nodes), ind=6)
    
    min_impurity_decrease = setParam('min_impurity_decrease', params, grid)
    info("Param: min_impurity_decrease = {0}".format(min_impurity_decrease), ind=6)
        
    min_samples_leaf = setParam('min_samples_leaf', params, grid)
    info("Param: min_samples_leaf = {0}".format(min_samples_leaf), ind=6)
    
    min_samples_split = setParam('min_samples_split', params, grid)
    info("Param: min_samples_split = {0}".format(min_samples_split), ind=6)
    
    min_weight_fraction_leaf = setParam('min_weight_fraction_leaf', params, grid)
    info("Param: min_weight_fraction_leaf = {0}".format(min_weight_fraction_leaf), ind=6)
    
    
    ## Estimator
    reg = DecisionTreeClassifier(criterion=criterion, 
                                max_depth=max_depth, max_features=max_features, 
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                min_weight_fraction_leaf=min_weight_fraction_leaf)
    
    return {"estimator": reg, "params": tuneParams}




def getDecisionTreeClassifierParams():
    treeParams = {"max_depth": [2, 4, 6, None],
                  "max_features": ['auto', 'sqrt', 'log2', None],
                  "min_impurity_decrease": rfloat(0.0, 0.25),
                  "min_samples_leaf": rint(1, 10)}
    
    params = treeParams
    params["criterion"] = ["gini", "entropy"]


    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval
