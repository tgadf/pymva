#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:08:08 2018

@author: tgadfort
"""

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, 
from sklearn.linear_model import LassoCV, LassoLarsCV, MultiTaskLasso
from sklearn.ensemble import RandomForestClassifier

def getParams(estimator, X_train):
    
    params = {}
    
    ## Generalized Linear Models
    if isinstance(estimator, LinearRegression):
        params = {}
    if isinstance(estimator, Ridge):
        params = {"alpha": [0, 1]}
    if isinstance(estimator, RidgeCV):
        # L_{2} = \alpha {||w||_2}^2}
        params = {"alphas": [0, 1]}
    if isinstance(estimator, Lasso):
        # L = (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        params = {"alpha": [0, 1]}
    if isinstance(estimator, LassoCV):
        # L = (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        params = {"alphas": [0, 1]}
    if isinstance(estimator, LassoLars):
        # L = (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        params = {"alpha": [0.1, 1]}
    if isinstance(estimator, LassoLarsCV):
        # L = (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        params = {}
    if isinstance(estimator, LassoLarsIC):
        # L = (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        params = {"criterion": ["bic", "aic"]}
    if isinstance(estimator, MultiTaskLasso):
        params = {"alphas": [0, 1]}
        # (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21
        # ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}


    #if isinstance(estimator, RandomForestClassifier):
    #    params = {"max_depth": [2, 10]}
        
    return params