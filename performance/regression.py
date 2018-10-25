#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:50:11 2018

@author: tgadfort
"""

from logger import info, error

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


        
def getPerformance(y_truth, testResults):
    info("Getting regression performance", ind=2)
    
    y_pred = testResults['pred']

    retval = {}
    
    evs = explained_variance_score(y_truth, y_pred)
    retval["ExplainedVariance"] = evs
    info("Explained Variance: {0}".format(round(evs, 2)), ind=6)
    
    mae = mean_absolute_error(y_truth, y_pred)  
    retval["MeanAbsoluteError"] = mae
    info("Mean Absolute Error: {0}".format(round(mae, 2)), ind=6)
    
    mse = mean_squared_error(y_truth, y_pred)  
    retval["MeanSquaredError"] = mse
    info("Mean Squared Error: {0}".format(round(mse, 2)), ind=6)
    
    #info("Getting mean squared log error", ind=6)
    #msle = mean_squared_log_error(y_truth, y_pred)  
    #retval["MeanSquaredLogError"] = msle
    
    mdae = median_absolute_error(y_truth, y_pred)  
    retval["MedianAbsoluteError"] = mdae
    info("Median Absolute Error: {0}".format(round(mdae, 2)), ind=6)
    
    r2s = r2_score(y_truth, y_pred)  
    retval["R2Score"] = r2s
    info("Getting r2 score: {0}".format(round(r2s, 2)), ind=6)
    
    res = y_truth - y_pred
    retval["Residuals"] = res
    info("Getting residuals: (mean,std): {0} , {1}".format(round(res.mean(), 2), round(res.std(), 2)), ind=6)
    
    return retval