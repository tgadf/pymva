#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:49:10 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen, genPowerTwo

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,RANSACRegressor




###############################################################################
#
# Linear Regression
#
#   Penalty: None
#
###############################################################################
def createLinearRegressor(params = None):
    info("Creating Linear Regressor", ind=4)
    
    ## Params
    params     = mergeParams(LinearRegression(), params)
    tuneParams = getLinearRegressorParams()
    info("Without Parameters", ind=4)

    ## estimator
    reg = LinearRegression()
    
    return {"estimator": reg, "params": tuneParams}
    


def getLinearRegressorParams():
    retval = {"dist": {}, "grid": {}}    
    return retval




###############################################################################
#
# Ridge Regression
#
#   Penalty: alpha * L2
#
###############################################################################
def createRidgeRegressor(params = None):
    info("Creating Ridge Regressor", ind=4)
    
    ## Params
    params     = mergeParams(RidgeCV(), params)
    params     = mergeParams(Ridge(), params)


    ## Estimator
    if params.get('cv') is True:
        info("Using Built-In Cross Validation With Parameters", ind=4)
        tuneParams = getRidgeRegressorParams(cv = True)
        grid       = tuneParams['grid']
        
        alphas = setParam('alphas', params, grid, force = True)
        info("Param: alphas = {0}".format(alphas), ind=6)
        
        reg = RidgeCV(alphas=alphas)
    else:
        info("With Parameters", ind=4)
        tuneParams = getRidgeRegressorParams(cv = False)
        grid       = tuneParams['grid']

        alpha = setParam('alpha', params, grid, force = False)
        info("Param: alpha = {0}".format(alpha), ind=6)
        
        reg = Ridge(alpha=alpha)
    
    return {"estimator": reg, "params": tuneParams}



def getRidgeRegressorParams(cv = False):
    param_grid = {}
    params = {}
    if cv is True:
        params["alphas"] = tuple(genPowerTen(-1, 1, 9))
    else:
        params["alpha"] = genPowerTen(-1, 1, 9)
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval
                
        

###############################################################################
#
# Lasso Regression
#
#   Penalty: alpha * L1
#
###############################################################################
def createLassoRegressor(params = None):
    info("Creating Lasso Regressor", ind=4)
    
    ## Params
    params     = mergeParams(LassoCV(), params)
    params     = mergeParams(Lasso(), params)


    ## Estimator
    if params.get('cv') is True:
        info("Using Built-In Cross Validation With Parameters", ind=4)
        tuneParams = getLassoRegressorParams(cv=True)
        grid       = tuneParams['grid']
        
        n_alphas = setParam('n_alphas', params, grid, force=True)
        info("Param: n_alphas = {0}".format(n_alphas), ind=6)
        
        reg = LassoCV(n_alphas=n_alphas)
    else:
        info("With Parameters", ind=4)
        tuneParams = getLassoRegressorParams(cv=False)
        grid       = tuneParams['grid']

        alpha = setParam('alpha', params, grid, force=False)
        info("Param: alpha = {0}".format(alpha), ind=6)
        
        reg = Lasso(alpha=alpha)
    
    return {"estimator": reg, "params": tuneParams}



def getLassoRegressorParams(cv = False):
    param_grid = {}
    params = {}
    if cv is True:
        params["n_alphas"] = 9
    else:
        params["alpha"] = genPowerTen(-1, 1, 9)
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# ElasticNet Regression
#
#   Penalty: a * L1 + b * L2
#            alpha = a + b and l1_ratio = a / (a + b)
#
###############################################################################
def createElasticNetRegressor(params = None):
    info("Creating ElasticNet Regressor", ind=4)
    
    ## Params
    params     = mergeParams(ElasticNetCV(), params)
    params     = mergeParams(ElasticNet(), params)


    ## Estimator
    if params.get('cv') is True:
        info("Using Built-In Cross Validation With Parameters", ind=4)
        tuneParams = getElasticNetRegressorParams(cv=True)
        grid       = tuneParams['grid']
        
        alphas = setParam('alphas', params, grid, force=True)
        info("Param: alphas = {0}".format(alphas), ind=6)

        l1_ratio = setParam('l1_ratio', params, grid, force=True)
        info("Param: l1_ratio = {0}".format(l1_ratio), ind=6)
        
        reg = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio)
    else:
        info("With Parameters", ind=4)
        tuneParams = getElasticNetRegressorParams(cv=False)
        grid       = tuneParams['grid']

        alpha = setParam('alpha', params, grid, force=False)
        info("Param: alpha = {0}".format(alpha), ind=6)
        
        l1_ratio = setParam('l1_ratio', params, grid, force=False)
        info("Param: l1_ratio = {0}".format(l1_ratio), ind=6)
        
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    
    return {"estimator": reg, "params": tuneParams}



def getElasticNetRegressorParams(cv=False):
    param_grid = {}
    if cv is True:
        params = {"l1_ratio": genPowerTen(-2, 0, 9),
                  "alphas": genPowerTen(-1, 1, 9)}
    else:
        params = {"l1_ratio": genPowerTen(-2, 0, 9),
                  "alpha": genPowerTen(-1, 1, 9)}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# OrthogonalMatchingPursuit
#
#   Penalty: ???
#
###############################################################################
def createOrthogonalMatchingPursuitRegressor(params = None):
    info("Creating Orthogonal Matching Pursuit Regressor", ind=4)
    
    ## Params
    params     = mergeParams(OrthogonalMatchingPursuit(), params)
    params     = mergeParams(OrthogonalMatchingPursuitCV(), params)
    tuneParams = getOrthogonalMatchingPursuitRegressorParams()

    ## estimator
    if params.get('cv') is True:
        info("Using Built-In Cross Validation With Parameters", ind=4)
        reg = OrthogonalMatchingPursuitCV()
    else:
        info("Without Parameters", ind=4)
        reg = OrthogonalMatchingPursuit()
    
    return {"estimator": reg, "params": tuneParams}
    


def getOrthogonalMatchingPursuitRegressorParams():
    retval = {"dist": {}, "grid": {}}    
    return retval



###############################################################################
#
# Bayesian Ridge Regression
#
#   Penalty: ???
#
###############################################################################
def createBayesianRidgeRegressor(params = None):
    info("Creating Bayesian Ridge Regressor", ind=4)
    
    ## Params
    params     = mergeParams(BayesianRidge(), params)
    tuneParams = getBayesianRidgeRegressorParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    alpha_1 = setParam('alpha_1', params, grid, force=False)
    info("Param: alpha_1 = {0}".format(alpha_1), ind=6)
    
    lambda_1 = setParam('lambda_1', params, grid, force=False)
    info("Param: lambda_1 = {0}".format(lambda_1), ind=6)
        
    alpha_2 = setParam('alpha_2', params, grid, force=False)
    info("Param: alpha_2 = {0}".format(alpha_2), ind=6)
    
    lambda_2 = setParam('lambda_2', params, grid, force=False)
    info("Param: lambda_2 = {0}".format(lambda_2), ind=6)

    ## estimator
    reg = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2,
                        lambda_1=lambda_1, lambda_2=lambda_2)
    
    return {"estimator": reg, "params": tuneParams}
    


def getBayesianRidgeRegressorParams():
    param_grid = {}
    params = {"alpha_1": genPowerTen(-8, -4, 9),
              "lambda_1": genPowerTen(-8, -2, 13),
              "alpha_2": genPowerTen(-8, -4, 9),
              "lambda_2": genPowerTen(-8, -2, 13)}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# ARD Regression
#
#   Penalty: ???
#
###############################################################################
def createARDRegressor(params = None):
    info("Creating ARD Regressor", ind=4)
    
    ## Params
    params     = mergeParams(ARDRegression(), params)
    tuneParams = getARDRegressorParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    alpha_1 = setParam('alpha_1', params, grid, force=False)
    info("Param: alpha_1 = {0}".format(alpha_1), ind=6)
    
    lambda_1 = setParam('lambda_1', params, grid, force=False)
    info("Param: lambda_1 = {0}".format(lambda_1), ind=6)
        
    alpha_2 = setParam('alpha_2', params, grid, force=False)
    info("Param: alpha_2 = {0}".format(alpha_2), ind=6)
    
    lambda_2 = setParam('lambda_2', params, grid, force=False)
    info("Param: lambda_2 = {0}".format(lambda_2), ind=6)
        

    ## estimator
    reg = ARDRegression(alpha_1=alpha_1, alpha_2=alpha_2,
                        lambda_1=lambda_1, lambda_2=lambda_2)
    
    return {"estimator": reg, "params": tuneParams}
    


def getARDRegressorParams():
    param_grid = {}
    params = {"alpha_1": genPowerTen(-7, -5, 3),
              "lambda_1": genPowerTen(-7, -5, 3),
              "alpha_2": genPowerTen(-7, -5, 3),
              "lambda_2": genPowerTen(-7, -5, 3)}
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
def createSGDRegressor(params):
    info("Creating SGD Regressor", ind=4)
    
    ## Params
    params     = mergeParams(SGDRegressor(), params)
    tuneParams = getSGDRegressorParams()
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
    reg = SGDRegressor(alpha=alpha, epsilon=epsilon, eta0=eta0, 
                        l1_ratio=l1_ratio, learning_rate=learning_rate, 
                        loss=loss, penalty=penalty,
                        power_t=power_t)
    
    return {"estimator": reg, "params": tuneParams}


def getSGDRegressorParams():
    param_grid = {}
    params = {"alpha": genPowerTen(-1, 1, 4),
              "epsilon": genLinear(0.05, 0.25, step=0.05),
              "eta0": genPowerTen(-3, -1, 5),
              "l1_ratio": genPowerTen(-2, 0, 5),
              "learning_rate": ["constant", "optimal", "invscaling"],
              "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
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
def createPassiveAggressiveRegressor(params):
    info("Creating Passive Aggressive Regressor", ind=4)
    
    ## Params
    params     = mergeParams(PassiveAggressiveRegressor(), params)
    tuneParams = getPassiveAggressiveRegressorParams()
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
    reg = PassiveAggressiveRegressor(C=C, loss=loss, max_iter=max_iter, tol=tol)
    
    return {"estimator": reg, "params": tuneParams}



def getPassiveAggressiveRegressorParams():
    param_grid = {}
    params = {"C": genPowerTen(-1, 1, 9),
              "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
              "max_iter": [1000],
              "tol": [0.001]}
                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# Huber Robust Regression
#
#   Penalty: ???
#
###############################################################################
def createHuberRegressor(params):
    info("Creating Huber Regressor", ind=4)
    
    ## Params
    params     = mergeParams(HuberRegressor(), params)
    tuneParams = getHuberRegressorParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    alpha = setParam('alpha', params, grid, force=False)
    info("Param: alpha = {0}".format(alpha), ind=6)

    epsilon = setParam('epsilon', params, grid, force=False)
    info("Param: epsilon = {0}".format(epsilon), ind=6)

    max_iter = setParam('max_iter', params, grid, force=False)
    info("Param: max_iter = {0}".format(max_iter), ind=6)

    tol = setParam('tol', params, grid, force=False)
    info("Param: tol = {0}".format(tol), ind=6)


    # estimator
    reg = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=max_iter, tol=tol)
    
    return {"estimator": reg, "params": tuneParams}


def getHuberRegressorParams():
    param_grid = {}
    params = {"alpha": genPowerTen(-5, -3, 9),
              "epsilon": genLinear(1.05, 1.65, step=0.05),
              "max_iter": [1000],
              "tol": [0.00001]}
                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# TheilSen Robust Regression
#
#   Penalty: ???
#
###############################################################################
def createTheilSenRegressor(params):
    info("Creating TheilSen Regressor", ind=4)
    
    ## Params
    params     = mergeParams(TheilSenRegressor(), params)
    tuneParams = getTheilSenRegressorParams()
    info("Without Parameters", ind=4)


    ## estimator
    reg = TheilSenRegressor()
    
    return {"estimator": reg, "params": tuneParams}


def getTheilSenRegressorParams():
    param_grid = {}
    params = {}                            
    retval = {"dist": params, "grid": param_grid}
    return retval



###############################################################################
#
# RANSAC Regression
#
#   Penalty: ???
#
###############################################################################
def createRANSACRegressor(params):
    info("Creating TheilSen Regressor", ind=4)
    
    ## Params
    params     = mergeParams(RANSACRegressor(), params)
    tuneParams = getRANSACRegressorParams()
    info("Without Parameters", ind=4)


    ## estimator
    reg = RANSACRegressor()
    
    return {"estimator": reg, "params": tuneParams}


def getRANSACRegressorParams():
    param_grid = {}
    params = {}                            
    retval = {"dist": params, "grid": param_grid}
    return retval