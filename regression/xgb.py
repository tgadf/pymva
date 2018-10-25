#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:07:28 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen

# conda install py-xgboost

from xgboost import XGBRegressor

from numpy import linspace, power

from scipy.stats import randint as rint
from scipy.stats import uniform as rfloat

###############################################################################
#
# XGBoost Regressor
#
###############################################################################
def createXgboostRegressor(params = None): 
    info("Creating Xgboost Regressor", ind=4)
    
    ## Params
    params     = mergeParams(XGBRegressor(), params)
    tuneParams = getXGBRegressorParams()
    grid       = tuneParams['grid']
    
    info("With Parameters", ind=6)
    gamma = setParam('gamma', params, grid)
    info("Param: gamma = {0}".format(gamma), ind=6)
        
    max_depth = setParam('max_depth', params, grid)
    info("Param: max_depth = {0}".format(max_depth), ind=6)
        
    learning_rate = setParam('learning_rate', params, grid)
    info("Param: learning_rate = {0}".format(learning_rate), ind=6)
        
    n_estimators = setParam('n_estimators', params, grid)
    info("Param: n_estimators = {0}".format(n_estimators), ind=6)
        
    nthread = setParam('nthread', params, grid)
    info("Param: nthread = {0}".format(nthread), ind=6)
        
    reg_alpha = setParam('reg_alpha', params, grid)
    info("Param: reg_alpha = {0}".format(reg_alpha), ind=6)
        
    reg_lambda = setParam('reg_lambda', params, grid)
    info("Param: reg_lambda = {0}".format(reg_lambda), ind=6)
    
    
    ## Estimator
    reg = XGBRegressor(gamma=gamma,
                       learning_rate=learning_rate, 
                       max_depth=max_depth,
                       n_estimators=n_estimators, nthread=nthread,
                       reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    
    return {"estimator": reg, "params": tuneParams}


def getXGBRegressorParams():
    params = {"gamma": genLinear(0, 1, step=0.2),
              "max_depth": genLinear(2, 8, step=2),
              "learning_rate": genPowerTen(-2, -0.5, 4),
              "n_estimators": [50, 100, 200, 350, 500],
              "reg_alpha": genPowerTen(-2, 1, 4),
              "reg_lambda": genPowerTen(-2, 1, 4)}

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
        
    retval = {"dist": params, "grid": param_grid}    
    return retval