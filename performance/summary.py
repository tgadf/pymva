#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:51:29 2018

@author: tgadfort
"""

from logger import info, error

from targetInfo import isClassification, isRegression
from classification import getPerformance as getClassifierPerformance
from regression import getPerformance as getRegressionPerformance

def getModelPerformance(y_truth, testResults, config):
    info("Getting model performance", ind=0)
    
    problemType = config['problem']
    
    if isClassification(problemType):
        try:
            results = getClassifierPerformance(y_truth, testResults)
        except:
            error("There was a problem getting classification performance data", ind=4)
            results = {}
        
    if isRegression(problemType):
        try:
            results = getRegressionPerformance(y_truth, testResults)
        except:
            error("There was a problem getting regression performance data", ind=4)
            results = {}
        
    return results
