#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:59:16 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import getParams

from lightning.classification import CDClassifier
from sklearn.datasets import fetch_20newsgroups_vectorized

###############################################################################
#
# Lightning Classifier
#
#   conda install -c conda-forge sklearn-contrib-lightning
#   https://github.com/scikit-learn-contrib/lightning
#
###############################################################################
def createLightningClassification(params = None): 
    ## Params
    lParams   = CDClassifier().get_params()
    if params is None:
        params = lParams

    C           = getParams('C', float, None, params, lParams)
    Cd          = getParams('Cd', float, None, params, lParams)
    alpha       = getParams('alpha', float, None, params, lParams)
    beta        = getParams('beta', float, None, params, lParams)
    loss        = getParams('loss', str, ['squared_hinge'], params, lParams)
    max_iter    = getParams('max_iter', int, None, params, lParams)
    max_steps   = getParams('max_steps', str, ['auto'], params, lParams)
    n_calls     = getParams('n_calls', int, None, params, lParams)
    n_jobs      = getParams('n_jobs', int, None, params, lParams)
    penalty     = getParams('penalty', str, ['l1', 'l2', 'l1/l2'], params, lParams)
    sigma       = getParams('sigma', float, None, params, lParams)
    termination = getParams('termination', str, ['violation_max', 'violation_sum'], params, lParams)
    tol         = getParams('tol', float, None, params, lParams)
 
 
        
    ## Estimator
    clf = CDClassifier(C=C, Cd=Cd, alpha=alpha, beta=beta,
                       loss=loss, max_iter=max_iter,
                       max_steps=max_steps, n_calls=n_calls,
                       n_jobs=n_jobs, penalty=penalty,
                       sigma=sigma, termination=termination, tol=tol)
    
    return clf
 
#
## Load News20 dataset from scikit-learn.
#bunch = fetch_20newsgroups_vectorized(subset="all")
#X = bunch.data
#y = bunch.target
#
## Set classifier options.
#clf = CDClassifier(penalty="l1/l2",
#                   loss="squared_hinge",
#                   multiclass=True,
#                   max_iter=20,
#                   alpha=1e-4,
#                   C=1.0 / X.shape[0],
#                   tol=1e-3)
#
## Train the model.
#clf.fit(X, y)
#
## Accuracy
#print(clf.score(X, y))
#
## Percentage of selected features
#print(clf.n_nonzero(percentage=True))