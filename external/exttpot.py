#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:15:45 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import getParams

from tpot import TPOTClassifier, TPOTRegressor

###############################################################################
#
# TPOT Classifier
#
#   pip install deap update_checker tqdm stopit
#   conda install py-xgboost
#   pip install scikit-mdr skrebate
#   pip install tpot
#
#   http://link.springer.com/chapter/10.1007/978-3-319-31204-0_9
#
###############################################################################
def createTPOTClassification(params = None):
    ## Params
    tpParams   = TPOTClassifier().get_params()
    if params is None:
        params = tpParams
    
    crossover_rate  = getParams('crossover_rate', float, None, params, tpParams)
    cv              = getParams('cv', int, None, params, tpParams)
    generations     = getParams('generations', int, None, params, tpParams)
    max_time_mins   = getParams('max_time_mins', (type(None), int), None, params, tpParams)
    mutation_rate   = getParams('mutation_rate', float, None, params, tpParams)
    n_jobs          = getParams('n_jobs', int, None, params, tpParams)
    population_size = getParams('population_size', int, None, params, tpParams)
    verbosity       = getParams('verbosity', int, None, params, tpParams)
    verbosity = 2
    cv = 2
    generations = 4
    max_time_mins = 100
        

    info("Creating TPOT Classifier with Parameters", ind=4)
    info("Param: crossover_rate = {0}".format(crossover_rate), ind=6)
    info("Param: cv = {0}".format(cv), ind=6)
    info("Param: generations = {0}".format(generations), ind=6)
    info("Param: max_time_mins = {0}".format(max_time_mins), ind=6)
    info("Param: mutation_rate = {0}".format(mutation_rate), ind=6)
    info("Param: n_jobs = {0}".format(n_jobs), ind=6)
    info("Param: population_size = {0}".format(population_size), ind=6)
    info("Param: verbosity = {0}".format(verbosity), ind=6)
    clf = TPOTClassifier(crossover_rate=crossover_rate, cv=cv, 
                         generations=generations,
                         n_jobs=n_jobs, mutation_rate=mutation_rate,
                         population_size=population_size, 
                         verbosity=verbosity)

    return clf
    
    
    
    

###############################################################################
#
# TPOT Regressor
#
###############################################################################
def createTPOTRegression(params = None):
    ## Params
    tpParams   = TPOTRegressor().get_params()
    if params is None:
        params = tpParams
    
    crossover_rate  = getParams('crossover_rate', float, None, params, tpParams)
    cv              = getParams('cv', int, None, params, tpParams)
    generations     = getParams('generations', int, None, params, tpParams)
    max_time_mins   = getParams('max_time_mins', (type(None), int), None, params, tpParams)
    mutation_rate   = getParams('mutation_rate', float, None, params, tpParams)
    n_jobs          = getParams('n_jobs', int, None, params, tpParams)
    population_size = getParams('population_size', int, None, params, tpParams)
    verbosity       = getParams('verbosity', int, None, params, tpParams)
    verbosity = 2
    cv = 2
    generations = 4
    max_time_mins = 100
        

    info("Creating TPOT Classifier with Parameters", ind=4)
    info("Param: crossover_rate = {0}".format(crossover_rate), ind=6)
    info("Param: cv = {0}".format(cv), ind=6)
    info("Param: generations = {0}".format(generations), ind=6)
    info("Param: max_time_mins = {0}".format(max_time_mins), ind=6)
    info("Param: mutation_rate = {0}".format(mutation_rate), ind=6)
    info("Param: n_jobs = {0}".format(n_jobs), ind=6)
    info("Param: population_size = {0}".format(population_size), ind=6)
    info("Param: verbosity = {0}".format(verbosity), ind=6)
    clf = TPOTClassifier(crossover_rate=crossover_rate, cv=cv, 
                         generations=generations,
                         n_jobs=n_jobs, mutation_rate=mutation_rate,
                         population_size=population_size, 
                         verbosity=verbosity)

    return clf



###############################################################################
#
# Export TPOT
#
###############################################################################
def exportTPOT(tpotObj, config):
    name = "dummy_tpot.py"
    info("Exporting TPOT object", ind=2)
    try:
        tpotObj.export(name)
    except:
        raise ValueError("Could not export TPOT object to",name)