#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:04:55 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import randint as rint
from scipy.stats import uniform as rfloat


###############################################################################
#
# Random Forest Regressor
#
###############################################################################
def createRandomForestRegressor(params):
    info("Creating Random Forest Regressor", ind=4)
    
    ## Params
    params     = mergeParams(RandomForestRegressor(), params)
    tuneParams = getRandomForestRegressorParams()
    grid       = tuneParams['grid']

    
    info("With Parameters", ind=6)
    bootstrap = setParam('bootstrap', params, grid)
    info("Param: bootstrap = {0}".format(bootstrap), ind=6)
        
    criterion = setParam('criterion', params, grid)
    info("Param: criterion = {0}".format(criterion), ind=6)
        
    max_depth = setParam('max_depth', params, grid)
    info("Param: max_depth = {0}".format(max_depth), ind=6)
    
    max_features = setParam('max_features', params, grid)
    info("Param: max_features = {0}".format(max_features), ind=6)
    
    min_impurity_decrease = setParam('min_impurity_decrease', params, grid)
    info("Param: min_impurity_decrease = {0}".format(min_impurity_decrease), ind=6)
        
    min_samples_leaf = setParam('min_samples_leaf', params, grid)
    info("Param: min_samples_leaf = {0}".format(min_samples_leaf), ind=6)
    
    n_estimators = setParam('n_estimators', params, grid)
    info("Param: n_estimator = {0}".format(n_estimators), ind=6)
        
    n_jobs = setParam('n_jobs', params, grid)
    n_jobs = -1
    info("Param: n_jobs = {0}".format(n_jobs), ind=6)
        
        
    ## Estimator
    reg = RandomForestRegressor(bootstrap=bootstrap, criterion=criterion, 
                                max_depth=max_depth, max_features=max_features, 
                                min_impurity_decrease=min_impurity_decrease,
                                min_samples_leaf=min_samples_leaf,
                                n_estimators=n_estimators, n_jobs=n_jobs)
    
    return {"estimator": reg, "params": tuneParams}



def getRandomForestRegressorParams():
    treeParams = {"max_depth": [2, 4, 6, 8, None],
                  "max_features": ['auto', 'sqrt', 'log2', None],
                  "min_impurity_decrease": rfloat(0.0, 0.25),
                  "min_samples_leaf": rint(1, 10)}
    
    params = treeParams
    #params = {}
    #params["bootstrap"] = [False, True]
    #params["criterion"] = ["mae", "mse"]
    params["n_estimators"] = [50] #rint(10, 100) #[50, 100]#, 200, 350, 500]

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval



###############################################################################
#
# Extra Trees Regressor
#
###############################################################################
def createExtraTreesRegressor(params):
    info("Creating Extra Trees Regressor", ind=4)
    
    ## Params
    params     = mergeParams(ExtraTreesRegressor(), params)
    tuneParams = getExtraTreesRegressorParams()
    grid       = tuneParams['grid']

    
    info("With Parameters", ind=6)
    bootstrap = setParam('bootstrap', params, grid)
    info("Param: bootstrap = {0}".format(bootstrap), ind=6)
        
    criterion = setParam('criterion', params, grid)
    info("Param: criterion = {0}".format(criterion), ind=6)
        
    max_depth = setParam('max_depth', params, grid)
    info("Param: max_depth = {0}".format(max_depth), ind=6)
    
    max_features = setParam('max_features', params, grid)
    info("Param: max_features = {0}".format(max_features), ind=6)
    
    min_impurity_decrease = setParam('min_impurity_decrease', params, grid)
    info("Param: min_impurity_decrease = {0}".format(min_impurity_decrease), ind=6)
        
    min_samples_leaf = setParam('min_samples_leaf', params, grid)
    info("Param: min_samples_leaf = {0}".format(min_samples_leaf), ind=6)
    
    n_estimators = setParam('n_estimators', params, grid)
    info("Param: n_estimator = {0}".format(n_estimators), ind=6)
        
    n_jobs = setParam('n_jobs', params, grid)
    n_jobs = -1
    info("Param: n_jobs = {0}".format(n_jobs), ind=6)
        
    
    ## Estimator
    reg = ExtraTreesRegressor(bootstrap=bootstrap, criterion=criterion, 
                              max_depth=max_depth, max_features=max_features, 
                              min_impurity_decrease=min_impurity_decrease,
                              min_samples_leaf=min_samples_leaf,
                              n_estimators=n_estimators, n_jobs=n_jobs)
    
    return {"estimator": reg, "params": tuneParams}
    


def getExtraTreesRegressorParams():
    treeParams = {"max_depth": [2, 4, 6, 8, None],
                  "max_features": ['auto', 'sqrt', 'log2', None],
                  "min_impurity_decrease": rfloat(0.0, 0.25),
                  "min_samples_leaf": rint(1, 10)}
    
    params = treeParams
    #params["bootstrap"] = [False, True]
    #params["criterion"] = ["mae", "mse"]
    params["n_estimators"] = [100]

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval




###############################################################################
#
# AdaBoost Regressor
#
###############################################################################
def createAdaBoostRegressor(params):
    info("Creating AdaBoost Regressor", ind=4)
    
    ## Params
    params     = mergeParams(AdaBoostRegressor(), params)
    tuneParams = getAdaBoostRegressorParams()
    grid       = tuneParams['grid']
        
        
    info("With Parameters", ind=6)
    learning_rate = setParam('learning_rate', params, grid)
    info("Param: learning_rate = {0}".format(learning_rate), ind=6)

    n_estimators = setParam('n_estimators', params, grid)
    info("Param: n_estimator = {0}".format(n_estimators), ind=6)
        
    
    ## Estimator
    reg = AdaBoostRegressor(learning_rate=learning_rate,
                            n_estimators=n_estimators)
    
    return {"estimator": reg, "params": tuneParams}


def getAdaBoostRegressorParams():
    params={}
    params["learning_rate"] = genPowerTen(-2, 1, 5)
    params["n_estimators"]  = [100]

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval



###############################################################################
#
# Gradient Boosting Regressor
#
###############################################################################
def createGBMRegressor(params):
    info("Creating GBM Regressor", ind=4)
    
    ## Params
    params     = mergeParams(GradientBoostingRegressor(), params)
    tuneParams = getGradientBoostingRegressorParams()
    grid       = tuneParams['grid']
    
    info("With Parameters", ind=6)
    criterion = setParam('criterion', params, grid)
    info("Param: criterion = {0}".format(criterion), ind=6)
        
    learning_rate = setParam('learning_rate', params, grid)
    info("Param: learning_rate = {0}".format(learning_rate), ind=6)

    loss = setParam('loss', params, grid)
    info("Param: loss = {0}".format(loss), ind=6)

    max_depth = setParam('max_depth', params, grid)
    info("Param: max_depth = {0}".format(max_depth), ind=6)
    
    max_features = setParam('max_features', params, grid)
    info("Param: max_features = {0}".format(max_features), ind=6)
    
    min_impurity_decrease = setParam('min_impurity_decrease', params, grid)
    info("Param: min_impurity_decrease = {0}".format(min_impurity_decrease), ind=6)
        
    min_samples_leaf = setParam('min_samples_leaf', params, grid)
    info("Param: min_samples_leaf = {0}".format(min_samples_leaf), ind=6)
    
    n_estimators = setParam('n_estimators', params, grid)
    info("Param: n_estimator = {0}".format(n_estimators), ind=6)
    
        
    ## Estimator
    reg = GradientBoostingRegressor(criterion=criterion,
                                    learning_rate=learning_rate, loss=loss,
                                    max_depth=max_depth, max_features=max_features, 
                                    min_impurity_decrease=min_impurity_decrease,
                                    min_samples_leaf=min_samples_leaf,
                                    n_estimators=n_estimators)
    
    return {"estimator": reg, "params": tuneParams}


def getGradientBoostingRegressorParams():
    treeParams = {"max_depth": [2, 4, 6, 8]}
#                  "max_features": ['auto', 'sqrt', 'log2', None],
#                  "min_impurity_decrease": genLinear(0, 0.25, step=0.05),
#                  "min_samples_leaf": genLinear(1, 10, step=1)}
    
    params = treeParams
    #params["criterion"] = ["mae", "friedman_mse"]
    params["loss"] = ["ls"]
    #params = {}
    params["learning_rate"] = genPowerTen(-2, -0.5, 4)
    params["n_estimators"] = [50]

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval
