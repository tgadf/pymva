#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:09:37 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen

from sklearn.neural_network import MLPClassifier



###############################################################################
#
# Multi Layer Perceptron Classifier
#
###############################################################################
def createMLPClassifier(params = None):
    info("Creating MLP Classifier", ind=4)
    
    ## Params
    params     = mergeParams(MLPClassifier(), params)
    tuneParams = getMLPClassifierParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    activation = setParam('activation', params, grid, force=False)
    info("Param: activation = {0}".format(activation), ind=6)
    
    alpha = setParam('alpha', params, grid, force=False)
    info("Param: alpha = {0}".format(alpha), ind=6)
    
    alpha = setParam('alpha', params, grid, force=False)
    info("Param: alpha = {0}".format(alpha), ind=6)
    
    beta_1 = setParam('beta_1', params, grid, force=False)
    info("Param: beta_1 = {0}".format(beta_1), ind=6)
    
    beta_2 = setParam('beta_2', params, grid, force=False)
    info("Param: beta_2 = {0}".format(beta_2), ind=6)
    
    hidden_layer_sizes = setParam('hidden_layer_sizes', params, grid, force=False)
    info("Param: hidden_layer_sizes = {0}".format(hidden_layer_sizes), ind=6)
    
    learning_rate = setParam('learning_rate', params, grid, force=False)
    info("Param: learning_rate = {0}".format(learning_rate), ind=6)
    
    max_iter = setParam('max_iter', params, grid, force=False)
    info("Param: max_iter = {0}".format(max_iter), ind=6)
    
    momentum = setParam('momentum', params, grid, force=False)
    info("Param: momentum = {0}".format(momentum), ind=6)
    
    power_t = setParam('power_t', params, grid, force=False)
    info("Param: power_t = {0}".format(power_t), ind=6)
    
    solver = setParam('solver', params, grid, force=False)
    info("Param: solver = {0}".format(solver), ind=6)

    reg = MLPClassifier(activation=activation, alpha=alpha, beta_1=beta_1,
                       beta_2=beta_2, hidden_layer_sizes=hidden_layer_sizes,
                       learning_rate=learning_rate, max_iter=max_iter,
                       momentum=momentum,
                       power_t=power_t, solver=solver)
    
    return {"estimator": reg, "params": tuneParams}
    


def getMLPClassifierParams():
    param_grid = {}
    params = {"activation": ["identity", "logistic", "tanh", "relu"],
              "alpha": genPowerTen(-1, 1, 9),
              "beta_1": genLinear(0.81, 0.99, step=0.04),
              "hidden_layer_sizes": [(10,), (25,), (50,)], #, (100,), (250,)],
              #"learning_rate": ["constant", "invscaling", "adaptive"],
              #"momentum": genLinear(0.75, 0.95, step=0.05),
              "max_iter": [500]}
              #"power_t": genLinear(0.25, 0.75, step=0.25)}
              #"solver": ["lbfgs", "sgd", "adam"]}
              
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval