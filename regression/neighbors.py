#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:43:04 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor


###############################################################################
#
# Nearest Neighbor Regressor
#
###############################################################################
def createKNeighborsRegression(params):
    error("This doesn't work")
    return {"estimator": None, "params": None}
    ## Params
    nnParams   = KNeighborsRegressor().get_params()
    if params is None:
        params = nnParams
        
        
    algorithm = getParams('algorithm', str, ['auto', 'ball_tree', 'kd_tree', 'brute'], params, nnParams)
    leaf_size = getParams('leaf_size', int, None, params, nnParams)
    metric = getParams('metric', str, ['minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'], params, nnParams)
    n_jobs = getParams('n_jobs', int, None, params, nnParams)
    n_neighbors = getParams('n_neighbors', int, None, params, nnParams)
    weights = getParams('weights', str, ['uniform', 'distance'], params, nnParams)



    ## Estimator
    info("Creating KNeighbors Regressor with Parameters", ind=4)
    info("Param: algorithm = {0}".format(algorithm), ind=6)
    info("Param: leaf_size = {0}".format(leaf_size), ind=6)
    info("Param: metric = {0}".format(metric), ind=6)
    info("Param: n_jobs = {0}".format(n_jobs), ind=6)
    info("Param: n_neighbors = {0}".format(n_neighbors), ind=6)
    info("Param: weights = {0}".format(weights), ind=6)
    reg = KNeighborsRegressor(algorithm=algorithm, leaf_size=leaf_size, 
                              metric=metric, n_jobs=n_jobs, 
                              n_neighbors=n_neighbors, weights=weights)
    return reg


###############################################################################
#
# Radius Neighbor Regressor
#
###############################################################################
def createRadiusNeighborsRegressor(params = None):
    info("Creating Radius Neighbors Regressor", ind=4)
    error("This doesn't work")
    return {"estimator": None, "params": None}
    
    ## Params
    params     = mergeParams(RadiusNeighborsRegressor(), params)
    tuneParams = getRadiusNeighborsRegressorParams()
    grid       = tuneParams['grid']


    info("With Parameters", ind=4)
    algorithm = setParam('algorithm', params, grid, force=False)
    info("Param: algorithm = {0}".format(algorithm), ind=6)
        
    leaf_size = setParam('leaf_size', params, grid, force=False)
    info("Param: leaf_size = {0}".format(leaf_size), ind=6)
        
    metric = setParam('metric', params, grid, force=False)
    info("Param: metric = {0}".format(metric), ind=6)
        
    radius = setParam('radius', params, grid, force=False)
    info("Param: radius = {0}".format(radius), ind=6)
        
    weights = setParam('weights', params, grid, force=False)
    info("Param: weights = {0}".format(weights), ind=6)



    ## Estimator
    reg = RadiusNeighborsRegressor(algorithm=algorithm, leaf_size=leaf_size,
                                   metric=metric, radius=radius,weights=weights)
    
    return {"estimator": reg, "params": tuneParams}



def getRadiusNeighborsRegressorParams():
    params = {"algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
              "leaf_size": genLinear(10, 50, step=10),
              "metric": ['minkowski', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
              "radius": genLinear(0.5, 1.5, step=0.5),
              "weights": ['uniform', 'distance']}

    param_grid = {}
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
        
    retval = {"dist": params, "grid": param_grid}    
    return retval