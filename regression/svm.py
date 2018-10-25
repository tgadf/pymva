#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:25:16 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen, genPowerTwo

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, NuSVR, LinearSVR


###############################################################################
#
# Linear Support Vector Regressor
#
###############################################################################
def createSVMLinearRegressor(params = None):
    info("Creating SVM Linear Regressor", ind=4)
    
    ## Params
    params     = mergeParams(LinearSVR(), params)
    tuneParams = getSVMLinearRegressorParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)
    
    loss = setParam('loss', params, grid, force=False)
    info("Param: loss = {0}".format(loss), ind=6)
    
    
    ## estimator
    reg = LinearSVR(C=C, loss=loss)

    return {"estimator": reg, "params": tuneParams}


def getSVMLinearRegressorParams():
    param_grid = {}
    params = {"C": genPowerTen(-1, 1, 9),
              "loss": ['epsilon_insensitive', 'squared_epsilon_insensitive']}
              
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval





###############################################################################
#
# Nu Support Vector Regressor
#
###############################################################################
def createSVMNuPolyRegressor(params = None):
    info("Creating SVM Nu Poly Regressor", ind=4)
    
    ## Params
    params     = mergeParams(NuSVR(), params)
    kernel     = 'poly'
    tuneParams = getSVMNuRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)

    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    degree = setParam('degree', params, grid, force=False)
    info("Param: degree = {0}".format(degree), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    
    ## estimator
    reg = NuSVR(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel, nu=nu)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMNuLinearRegressor(params = None):
    info("Creating SVM Nu Poly Regressor", ind=4)
    
    ## Params
    params     = mergeParams(NuSVR(), params)
    kernel     = 'linear'
    tuneParams = getSVMNuRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)

    
    ## estimator
    reg = NuSVR(C=C, kernel=kernel, nu=nu)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMNuSigmoidRegressor(params = None):
    info("Creating SVM Nu Poly Regressor", ind=4)
    
    ## Params
    params     = mergeParams(NuSVR(), params)
    kernel     = 'sigmoid'
    tuneParams = getSVMNuRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    
    ## estimator
    reg = NuSVR(C=C, kernel=kernel, nu=nu, coef0=coef0, gamma=gamma)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMNuRbfRegressor(params = None):
    info("Creating SVM Nu Rbf Regressor", ind=4)
    
    ## Params
    params     = mergeParams(NuSVR(), params)
    kernel     = 'rbf'
    tuneParams = getSVMNuRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    
    ## estimator
    reg = NuSVR(C=C, kernel=kernel, nu=nu, gamma=gamma)
    
    return {"estimator": reg, "params": tuneParams}



def getSVMNuRegressorParams(kernel):
    param_grid = {}
    baseParams = {"C": genPowerTen(-1, 1, 9),
                  "nu": genLinear(0.1, 0.9, step=0.2),
                  "kernel": [kernel]}
    if kernel == "poly":
        params = {"coef0": genLinear(-1, 1, 3),
                  "degree": genLinear(1, 5, step=1),
                  "gamma": ['auto']}
    if kernel == "linear":
        params = {}
    if kernel == "sigmoid":
        params = {"coef0": genLinear(-1, 1, 3),
                  "gamma": ['auto']}
    if kernel == "rbf":
        params = {"gamma": ['auto']}
        
    params = dict(baseParams.items() + params.items())
              
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval





###############################################################################
#
# Epsilon Support Vector Regressor
#
###############################################################################
def createSVMEpsPolyRegressor(params = None):
    info("Creating SVM Epsilon Poly Regressor", ind=4)
    
    ## Params
    params     = mergeParams(SVR(), params)
    kernel     = 'poly'
    tuneParams = getSVMEpsRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    epsilon = setParam('epsilon', params, grid, force=False)
    info("Param: epsilon = {0}".format(epsilon), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)

    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    degree = setParam('degree', params, grid, force=False)
    info("Param: degree = {0}".format(degree), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    
    ## estimator
    reg = SVR(C=C, coef0=coef0, degree=degree, epsilon=epsilon,
              gamma=gamma, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMEpsLinearRegressor(params = None):
    info("Creating SVM Epsilon Poly Regressor", ind=4)
    
    ## Params
    params     = mergeParams(SVR(), params)
    kernel     = 'linear'
    tuneParams = getSVMEpsRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    epsilon = setParam('epsilon', params, grid, force=False)
    info("Param: epsilon = {0}".format(epsilon), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)

    
    ## estimator
    reg = SVR(C=C, epsilon=epsilon, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMEpsSigmoidRegressor(params = None):
    info("Creating SVM Epsilon Poly Regressor", ind=4)
    
    ## Params
    params     = mergeParams(SVR(), params)
    kernel     = 'sigmoid'
    tuneParams = getSVMEpsRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    epsilon = setParam('epsilon', params, grid, force=False)
    info("Param: epsilon = {0}".format(epsilon), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    
    ## estimator
    reg = SVR(C=C, coef0=coef0, epsilon=epsilon,
              gamma=gamma, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMEpsRbfRegressor(params = None):
    info("Creating SVM Epsilon Rbf Regressor", ind=4)
    
    ## Params
    params     = mergeParams(SVR(), params)
    kernel     = 'rbf'
    tuneParams = getSVMEpsRegressorParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    epsilon = setParam('epsilon', params, grid, force=False)
    info("Param: epsilon = {0}".format(epsilon), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    
    ## estimator
    reg = SVR(C=C, epsilon=epsilon,
              gamma=gamma, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}



def getSVMEpsRegressorParams(kernel):
    param_grid = {}
    baseParams = {"C": genPowerTen(-1, 1, 9),
                  "epsilon": genLinear(0.1, 0.9, step=0.2),
                  "kernel": [kernel]}
    if kernel == "poly":
        params = {"coef0": genLinear(-1, 1, 3),
                  "degree": genLinear(1, 5, step=1),
                  "gamma": ['auto']}
    if kernel == "linear":
        params = {}
    if kernel == "sigmoid":
        params = {"coef0": genLinear(-1, 1, 3),
                  "gamma": ['auto']}
    if kernel == "rbf":
        params = {"gamma": ['auto']}
        
    params = dict(baseParams.items() + params.items())
              
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# Kernal Ridge Regressor
#
#   Penalty: ???
#
###############################################################################
def createKernelRidgeRegressor(params = None):
    info("Creating SVM Regressor", ind=4)
    
    ## Params
    params     = mergeParams(KernelRidge(), params)
    tuneParams = getKernelRidgeRegressorParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    alpha = setParam('alpha', params, grid, force=False)
    info("Param: alpha = {0}".format(alpha), ind=6)
    
    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    degree = setParam('degree', params, grid, force=False)
    info("Param: degree = {0}".format(degree), ind=6)
        
    kernel = setParam('kernel', params, grid, force=False)
    info("Param: kernel = {0}".format(kernel), ind=6)


    ## estimator
    reg = KernelRidge(alpha=alpha, coef0=coef0, degree=degree, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}
    


def getKernelRidgeRegressorParams():
    param_grid = {}
    params = {"alpha": genPowerTen(-1, 1, 9),
              "coef0": genLinear(-1, 1, 3),
              "degree": genLinear(1, 5, step=1),
              "kernel": ['linear', 'poly', 'rbf', 'sigmoid']} #, 'precomputed']}
              
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval