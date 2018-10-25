#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:25:16 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC, NuSVC, LinearSVC


###############################################################################
#
# Linear Support Vector Classifier
#
###############################################################################
def createSVMLinearClassifier(params = None):
    info("Creating SVM Linear Classifier", ind=4)
    
    ## Params
    params     = mergeParams(LinearSVC(), params)
    tuneParams = getSVMLinearClassifierParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)
    
    loss = setParam('loss', params, grid, force=False)
    info("Param: loss = {0}".format(loss), ind=6)
    
    
    ## estimator
    reg = LinearSVC(C=C, loss=loss)

    return {"estimator": reg, "params": tuneParams}


def getSVMLinearClassifierParams():
    param_grid = {}
    params = {"C": genPowerTen(-1, 1, 9),
              "loss": ['epsilon_insensitive', 'squared_epsilon_insensitive']}
              
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval





###############################################################################
#
# Nu Support Vector Classifier
#
###############################################################################
def createSVMNuPolyClassifier(params = None):
    info("Creating SVM Nu Poly Classifier", ind=4)
    
    ## Params
    params     = mergeParams(NuSVC(), params)
    kernel     = 'poly'
    tuneParams = getSVMNuClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)

    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)

    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    degree = setParam('degree', params, grid, force=False)
    info("Param: degree = {0}".format(degree), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)
    
    ## estimator
    reg = NuSVC(coef0=coef0, degree=degree, gamma=gamma, 
                probability=probability, kernel=kernel, nu=nu)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMNuLinearClassifier(params = None):
    info("Creating SVM Nu Poly Classifier", ind=4)
    
    ## Params
    params     = mergeParams(NuSVC(), params)
    kernel     = 'linear'
    tuneParams = getSVMNuClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)

    
    ## estimator
    reg = NuSVC(kernel=kernel, nu=nu, probability=probability)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMNuSigmoidClassifier(params = None):
    info("Creating SVM Nu Poly Classifier", ind=4)
    
    ## Params
    params     = mergeParams(NuSVC(), params)
    kernel     = 'sigmoid'
    tuneParams = getSVMNuClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)
    
    
    ## estimator
    reg = NuSVC(kernel=kernel, nu=nu, coef0=coef0, 
                probability=probability, gamma=gamma)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMNuRbfClassifier(params = None):
    info("Creating SVM Nu Rbf Classifier", ind=4)
    
    ## Params
    params     = mergeParams(NuSVC(), params)
    kernel     = 'rbf'
    tuneParams = getSVMNuClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    nu = setParam('nu', params, grid, force=False)
    info("Param: nu = {0}".format(nu), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)
    
    
    ## estimator
    reg = NuSVC(kernel=kernel, nu=nu, 
                probability=probability, gamma=gamma)
    
    return {"estimator": reg, "params": tuneParams}



def getSVMNuClassifierParams(kernel):
    param_grid = {}
    baseParams = {"nu": genLinear(0.1, 0.9, step=0.2),
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
# Epsilon Support Vector Classifier
#
###############################################################################
def createSVMEpsPolyClassifier(params = None):
    info("Creating SVM Epsilon Poly Classifier", ind=4)
    
    ## Params
    params     = mergeParams(SVC(), params)
    kernel     = 'poly'
    tuneParams = getSVMEpsClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)

    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    degree = setParam('degree', params, grid, force=False)
    info("Param: degree = {0}".format(degree), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)
    
    
    ## estimator
    reg = SVC(C=C, coef0=coef0, degree=degree,
              gamma=gamma, probability=probability, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMEpsLinearClassifier(params = None):
    info("Creating SVM Epsilon Poly Classifier", ind=4)
    
    ## Params
    params     = mergeParams(SVC(), params)
    kernel     = 'linear'
    tuneParams = getSVMEpsClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)

    
    ## estimator
    reg = SVC(C=C, probability=probability, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMEpsSigmoidClassifier(params = None):
    info("Creating SVM Epsilon Poly Classifier", ind=4)
    
    ## Params
    params     = mergeParams(SVC(), params)
    kernel     = 'sigmoid'
    tuneParams = getSVMEpsClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    coef0 = setParam('coef0', params, grid, force=False)
    info("Param: coef0 = {0}".format(coef0), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)
    
    
    ## estimator
    reg = SVC(C=C, coef0=coef0, probability=probability,
              gamma=gamma, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}


def createSVMEpsRbfClassifier(params = None):
    info("Creating SVM Epsilon Rbf Classifier", ind=4)
    
    ## Params
    params     = mergeParams(SVC(), params)
    kernel     = 'rbf'
    tuneParams = getSVMEpsClassifierParams(kernel)
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    info("Param: kernel = {0}".format(kernel), ind=6)
    
    gamma = setParam('gamma', params, grid, force=False)
    info("Param: gamma = {0}".format(gamma), ind=6)
    
    probability = True
    info("Param: probability = {0}".format(probability), ind=6)
    
    
    ## estimator
    reg = SVC(C=C, probability=probability,
              gamma=gamma, kernel=kernel)
    
    return {"estimator": reg, "params": tuneParams}



def getSVMEpsClassifierParams(kernel):
    param_grid = {}
    baseParams = {"C": genPowerTen(-1, 1, 9),
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