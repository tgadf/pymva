#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:48:15 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genPowerTen

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

###############################################################################
#
# Gaussian Naive Bayes
#
###############################################################################
def createGaussianNaiveBayesClassifier(params):
    info("Creating Gaussian Naive Bayes Classifier", ind=4)
    
    ## Params
    params     = mergeParams(GaussianNB(), params)
    tuneParams = getGaussianNaiveBayesClassifierParams()
        

    info("Without Parameters", ind=4)        


    ## Estimator
    clf = GaussianNB()
    
    return {"estimator": clf, "params": tuneParams}


def getGaussianNaiveBayesClassifierParams():
    retval = {"dist": {}, "grid": {}}
    return retval


###############################################################################
#
# Bernoulli Naive Bayes
#
###############################################################################
def createBernoulliNaiveBayesClassifier(params):
    info("Creating Bernoulli Naive Bayes Classifier", ind=4)
    
    ## Params
    params     = mergeParams(BernoulliNB(), params)
    tuneParams = getBernoulliNaiveBayesClassifierParams()
    grid       = tuneParams['grid']
    
    
    info("With Parameters", ind=4)
    alpha = setParam('alpha', params, grid, force = False)
    info("Param: alpha = {0}".format(alpha), ind=6)
 

    ## Estimator    
    clf = BernoulliNB(alpha=alpha)
    
    return {"estimator": clf, "params": tuneParams}


def getBernoulliNaiveBayesClassifierParams():
    params = {"alpha": genPowerTen(-4, 4, 100)}

    param_grid = {}                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval


###############################################################################
#
# Multinomial Naive Bayes
#
###############################################################################
def createMultinomialNaiveBayesClassifier(params):
    info("Creating Multinomial Naive Bayes Classifier", ind=4)
    error("Multinomial Naive Bayes Classifier does not work", ind=4)
    return {"estimator": None, "params": None}
    
    ## Params
    params     = mergeParams(MultinomialNB(), params)
    tuneParams = getMultinomialNaiveBayesClassifierParams()
    grid       = tuneParams['grid']
    
    
    info("With Parameters", ind=4)
    alpha = setParam('alpha', params, grid, force = False)
    info("Param: alpha = {0}".format(alpha), ind=6)
 

    ## Estimator    
    clf = MultinomialNB(alpha=alpha)
    
    return {"estimator": clf, "params": tuneParams}


def getMultinomialNaiveBayesClassifierParams():
    params = {"alpha": genPowerTen(-4, 4, 100)}

    param_grid = {}                            
    for param,dist in params.iteritems():
        param_grid[param] = convertDistribution(dist)
    retval = {"dist": params, "grid": param_grid}
    return retval