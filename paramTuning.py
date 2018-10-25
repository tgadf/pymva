#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:23:08 2018

@author: tgadfort
"""

from logger import info

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import classification_report


from targetinfo import isClassification, isRegression, isClustering

def tuneParamsGrid(estimator, X_train, y_train, config):
    info("Tunning hyperparameters", ind=4)

    problemType  = config['problem']

    paramsConfig = config['params']
    searchType   = paramsConfig['type']
    cv           = paramsConfig['cv']
    scorer       = paramsConfig['scorer']
    grid         = paramsConfig.get('grid')
    dist         = paramsConfig.get('dist')
    
    if isClassification(problemType):
        scorers = ["accuracy", "average_precision", "f1", "f1_micro",
                   "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                   "precision", "recall", "roc_auc"]
    
    if isClustering(problemType):
        scorers = ["adjusted_mutual_info_score", "adjusted_rand_score",
                   "completeness_score", "fowlkes_mallows_score",
                   "homogeneity_score", "mutual_info_score",
                   "normalized_mutual_info_score", "v_measure_score"]
    
    if isRegression(problemType):
        scorers = ["explained_variance", "neg_mean_absolute_error",
                   "neg_mean_squared_error", "neg_mean_squared_log_error",
                   "neg_median_absolute_error", "r2"]
        
    if scorer not in scorers:
        raise ValueError("Scorer",scorer,"is not available for",problemType,"problem.")
        
    
    estimatorName = type(estimator).__name__
    if paramsConfig.get(estimatorName) == None:
        raise ValueError("There is no params grid for the",estimatorName,"estimator.")

    if searchType == "grid":
        param_grid = grid[estimatorName]
        clf = GridSearchCV(estimator, param_grid=param_grid, cv=cv,
                           scoring=scorer)
    elif searchType == "random":
        n_iter_search = 20
        if paramsConfig.get("iter"):
            n_iter_search = paramsConfig["iter"]
        param_dist = dist[estimatorName]
        clf = RandomizedSearchCV(estimator, param_distributions=param_dist,
                                   n_iter=n_iter_search)


    info("Running parameter search", ind=6)        
    clf.fit(X_train, y_train)

    bestEstimator = clf.best_estimator_        
    bestParams    = clf.best_params_
    cvResults     = clf.cv_results_
    
    info("Found best parameters for "+estimatorName)
    
    return clf