#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:15:42 2017

@author: tgadfort
"""

from logger import info
from targetInfo import getProblemType, isRegression, isClassification
from colInfo import isColumn, getDim
from convertData import convertToBinaryInt, convertCategoricalToNumeric
from fillNA import replaceTargetNA, replaceFeatureNA
from analyzeColumnData import dropData


def formatData(trainData, testData, config):
    info('Formatting training data of size '+getDim(trainData), ind=0)
    info('Formatting testing data of size '+getDim(testData), ind=0)
    
    
    ## Config info
    targetConfig      = config['target']
    targetcol         = targetConfig['colname']
    positiveTarget    = targetConfig['positive']
    targetNAstrategy  = targetConfig['NAstrategy']
    featureConfig     = config['feature']
    featureNAstrategy = featureConfig['NAstrategy']

    
    if not isColumn(trainData, targetcol):
        raise ValueError("Target column",targetcol,"is not a valid column.")
        

    # 1) Get problem type
    targetData  = trainData[targetcol]
    if config.get('problem'):
        problemType = config['problem']
    else:
        problemType = getProblemType(targetData)
        config['problem'] = problemType
    
    
    # 2) format target based on what we want
    info('Formatting target', ind=1)
    if isClassification(problemType):
        convertToBinaryInt(trainData, targetcol, positiveTarget)
        if isColumn(testData, targetcol):
            convertToBinaryInt(testData, targetcol, positiveTarget)        
    if isRegression(problemType):
        info('Not formatting target since it is regression', ind=1)
        
        
    # 3) replace NA
    info('Replace NA in data', ind=1)
    print featureNAstrategy
    replaceTargetNA(trainData, targetcol, targetNAstrategy)
    replaceFeatureNA(trainData, targetcol, featureNAstrategy)
    if isColumn(testData, targetcol):
        replaceTargetNA(testData, targetcol, targetNAstrategy)
    replaceFeatureNA(testData, targetcol, featureNAstrategy)
        

    # 4) drop columns we don't need
    dropData(trainData, config)
    dropData(testData, config)
    

    return trainData,testData
    
    # 5) format remaining data to numeric
    info('Formatting features to numeric', ind=1)
    convertCategoricalToNumeric(trainData, targetcol)
    convertCategoricalToNumeric(testData, targetcol)
    info('Post formatting the training data is now '+getDim(trainData), ind=2)
    info('Post formatting the testing data is now '+getDim(trainData), ind=2)
    
    #pddata.drop([colname], axis = 1, inplace = True)
    #pddata = pddata.join(expData)
            






    # 5) replace low variance
    info('Remove low variance features in data', ind=1)
    
    

    info('Finished formatting data', ind=0)
    
    return pddata