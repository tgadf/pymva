#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:41:45 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import mergeParams

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

###############################################################################
#
# Gaussian Process Classifier
#
###############################################################################
def createGaussianProcessRegressor(params = None):
    info("Creating Gaussian Process Regressor", ind=4)
    error("This takes forever. Don't use it")
    return {"estimator": None, "params": None}
    
    ## Params
    params     = mergeParams(GaussianProcessRegressor(), params)
    tuneParams = getGaussianProcessRegressorParams()


    info("Without Parameters", ind=4)
    kernel=kernels.ConstantKernel()
    

    ## Estimator
    reg = GaussianProcessRegressor(kernel=kernel)
        
    return {"estimator": reg, "params": tuneParams}




def getGaussianProcessRegressorParams():
    retval = {"dist": {}, "grid": {}}    
    return retval