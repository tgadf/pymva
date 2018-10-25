#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:41:45 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import mergeParams

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import kernels

###############################################################################
#
# Gaussian Process Classifier
#
###############################################################################
def createGaussianProcessClassifier(params = None):
    info("Creating Gaussian Process Classifier", ind=4)
    error("This takes forever. Don't use it")
    return {"estimator": None, "params": None}
    
    ## Params
    params     = mergeParams(GaussianProcessClassifier(), params)
    tuneParams = getGaussianProcessClassifierParams()


    info("Without Parameters", ind=4)
    kernel=kernels.ConstantKernel()
    

    ## Estimator
    reg = GaussianProcessClassifier(kernel=kernel)
        
    return {"estimator": reg, "params": tuneParams}




def getGaussianProcessClassifierParams():
    retval = {"dist": {}, "grid": {}}    
    return retval