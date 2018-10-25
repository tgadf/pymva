#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:21:32 2018

@author: tgadfort
"""

from logger import info

from paramHelper import mergeParams

from sklearn.isotonic import IsotonicRegression

###############################################################################
#
# Isotonic Regressor
#
###############################################################################
def createIsotonicRegressor(params = None):
    info("Creating Isotonic Regressor", ind=4)
    
    ## Params
    params     = mergeParams(IsotonicRegression(), params)
    tuneParams = getIsotonicRegressionParams()


    info("Without Parameters", ind=4)
    reg = IsotonicRegression(increasing="auto")
    
    return {"estimator": reg, "params": tuneParams}



def getIsotonicRegressionParams():
    retval = {"dist": {}, "grid": {}}    
    return retval
