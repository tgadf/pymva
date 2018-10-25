#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:28:27 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen, genPowerTwo

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


###############################################################################
#
# Logistic Regression
#
#   Penalty: l1: |w|1 + C * sum(i-N){log(exp(-y(XT*w)) + 1)}
#            l2: |w|2 + C * sum(i-N){log(exp(-y(XT*w)) + 1)}
#
###############################################################################
def createLogisticRegressionClassifier(params = None):
    info("Creating Logistic Regression Classifier", ind=4)
    
    ## Params
    params     = mergeParams(LogisticRegression(), params)
    params     = mergeParams(LogisticRegressionCV(), params)
    tuneParams = getLogisticRegressionClassifer()
    grid       = tuneParams['grid']
    
    
    ## Estimator
    if params.get('cv'):
        info("Using Built-In Cross Validation With Parameters", ind=4)
        tuneParams = getLogisticRegressionClassifer(cv = True)
        grid       = tuneParams['grid']
        
        Cs = setParam('Cs', params, grid, force = True)
        info("Param: Cs = {0}".format(Cs), ind=6)
        
        penalty = setParam('penalty', params, grid, force = True)
        info("Param: penalty = {0}".format(penalty), ind=6)
        
        solver = setParam('solver', params, grid, force = False)
        info("Param: solver = {0}".format(solver), ind=6)
        
        #n_jobs = -1
        #info("Param: n_jobs = {0}".format(n_jobs), ind=6)
        
        clf = LogisticRegressionCV(Cs=Cs, penalty=penalty, 
                                 solver=solver)
    else:
        info("With Parameters", ind=4)
        tuneParams = getLogisticRegressionClassifer(cv = False)
        grid       = tuneParams['grid']
        
        C = setParam('C', params, grid, force = False)
        info("Param: C = {0}".format(C), ind=6)
        
        penalty = setParam('penalty', params, grid, force = False)
        info("Param: penalty = {0}".format(penalty), ind=6)
        
        solver = setParam('solver', params, grid, force = False)
        info("Param: solver = {0}".format(solver), ind=6)
        
        #n_jobs = -1
        #info("Param: n_jobs = {0}".format(n_jobs), ind=6)
        
        clf = LogisticRegression(C=C, penalty=penalty, 
                                 solver=solver)
    
    return {"estimator": clf, "params": tuneParams}



def getLogisticRegressionClassifer(cv = False):
    param_grid = {}
    if cv is False:
        params = {"C": genPowerTen(-4, 4, 100),
                  "penalty": ["l1", "l2"]}
    else:
        params = {"Cs": genPowerTen(-2, 2, 100),
                  "penalty": ["l1", "l2"]}
                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# Stochastic Gradient Decent
#
#   Penalty: ???
#
###############################################################################
def createSGDClassifier(params):
    info("Creating SGD Classifier", ind=4)
    
    ## Params
    params     = mergeParams(SGDClassifier(), params)
    tuneParams = getSGDClassifierParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    alpha = setParam('alpha', params, grid, force=False)
    info("Param: alpha = {0}".format(alpha), ind=6)

    epsilon = setParam('epsilon', params, grid, force=False)
    info("Param: epsilon = {0}".format(epsilon), ind=6)

    eta0 = setParam('eta0', params, grid, force=False)
    info("Param: eta0 = {0}".format(eta0), ind=6)

    l1_ratio = setParam('l1_ratio', params, grid, force=False)
    info("Param: l1_ratio = {0}".format(l1_ratio), ind=6)

    learning_rate = setParam('learning_rate', params, grid, force=False)
    info("Param: learning_rate = {0}".format(learning_rate), ind=6)

    loss = setParam('loss', params, grid, force=False)
    info("Param: loss = {0}".format(loss), ind=6)

    max_iter = setParam('max_iter', params, grid, force=False)
    info("Param: max_iter = {0}".format(max_iter), ind=6)

    penalty = setParam('penalty', params, grid, force=False)
    info("Param: penalty = {0}".format(penalty), ind=6)

    power_t = setParam('power_t', params, grid, force=False)
    info("Param: power_t = {0}".format(power_t), ind=6)

    tol = setParam('tol', params, grid, force=False)
    info("Param: tol = {0}".format(tol), ind=6)

    
    ## Estimator
    clf = SGDClassifier(alpha=alpha, epsilon=epsilon, eta0=eta0, 
                        l1_ratio=l1_ratio, learning_rate=learning_rate, 
                        loss=loss, penalty=penalty,
                        power_t=power_t)
    
    return {"estimator": clf, "params": tuneParams}


def getSGDClassifierParams():
    param_grid = {}
    params = {"alpha": genPowerTen(-4, 4, 100),
              "epsilon": genLinear(0.05, 0.25, step=0.05),
              "eta0": genPowerTen(-3, -1, 5),
              "l1_ratio": genPowerTen(-2, 0, 5),
              "learning_rate": ["constant", "optimal", "invscaling"],
              "loss": ["modified_huber", "log"],
              "max_iter": [1000],
              "penalty": ["l1", "l2"],
              "power_t": genPowerTwo(-3,-1, 3),
              "tol": [0.001]}
                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval
    



###############################################################################
#
# PassiveAggressiveRegression
#
#   Penalty: ???
#
###############################################################################
def createPassiveAggressiveClassifier(params):
    info("Creating Passive Aggressive Classifier", ind=4)
    error("Does not give probabilities.")
    return {"estimator": None, "params": None}
    
    ## Params
    params     = mergeParams(PassiveAggressiveClassifier(), params)
    tuneParams = getPassiveAggressiveClassifierParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    C = setParam('C', params, grid, force=False)
    info("Param: C = {0}".format(C), ind=6)

    loss = setParam('loss', params, grid, force=False)
    info("Param: loss = {0}".format(loss), ind=6)

    max_iter = setParam('max_iter', params, grid, force=False)
    info("Param: max_iter = {0}".format(max_iter), ind=6)

    tol = setParam('tol', params, grid, force=False)
    info("Param: tol = {0}".format(tol), ind=6)


    ## Estimator
    clf = PassiveAggressiveClassifier(C=C, loss=loss, max_iter=max_iter, tol=tol)
    
    return {"estimator": clf, "params": tuneParams}



def getPassiveAggressiveClassifierParams():
    param_grid = {}
    params = {"C": genPowerTen(-1, 1, 9),
              "loss": ["hinge", "squared_hinge"],
              "max_iter": [1000],
              "tol": [0.001]}
                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval