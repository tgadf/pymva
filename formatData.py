#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:15:42 2017

@author: tgadfort
"""

from logger import info, error
from targetInfo import getProblemType, isRegression, isClassification
from colInfo import isColumn, getDim
from convertData import convertToBinaryInt, getEncodedData, getHotEncodedData
from fillNA import replaceTargetNA, replaceFeatureNA
from analyzeColumnData import dropData, analyzeColumns


def formatData(pddf, config):
    info('Formatting data of size '+getDim(pddf), ind=0)
    
    
    ## Config info
    targetConfig      = config['target']
    targetcol         = targetConfig['colname']
    positiveTarget    = targetConfig['positive']
    targetNAstrategy  = targetConfig['NAstrategy']
    featureConfig     = config['feature']
    featureNAstrategy = featureConfig['NAstrategy']

    
    if not isColumn(pddf, targetcol):
        raise ValueError("Target column",targetcol,"is not a valid column.")
        

    # 1) Get problem type
    targetData  = pddf[targetcol]
    if config.get('problem'):
        problemType = config['problem']
    else:
        problemType = getProblemType(targetData)
        config['problem'] = problemType
    
    
    # 2) format target based on what we want
    info('Formatting target', ind=2)
    if isClassification(problemType):
        convertToBinaryInt(pddf, targetcol, positiveTarget)
    if isRegression(problemType):
        info('Not formatting target since it is regression', ind=1)
        
        
    # 3) replace NA
    info('Replace NA in data', ind=2)
    replaceTargetNA(pddf, targetcol, targetNAstrategy)
    replaceFeatureNA(pddf, targetcol, featureNAstrategy)
    
    
    # 4) remove low variance data
    info('Remove low variance in data', ind=2)
    
        

    # 5) drop columns we don't need
    info('Analyze data for possible drops', ind=2)
    analyzeColumns(pddf, config)
    dropData(pddf, config)    
    info('Post column data the data is now '+getDim(pddf), ind=2)

    
    # 6) label and one-hot encode data
    info('Label encode training data to numeric', ind=2)
    pddf, encodedCatData, labelEncoders = getEncodedData(pddf)
    info('Hot encode training data to sparse data frame', ind=1)
    encodedData = getHotEncodedData(encodedCatData, labelEncoders)
    info('Join training data together', ind=2)
    pddf = pddf.join(encodedData)
    info('Post formatting the data is now '+getDim(pddf), ind=2)
            

    # 7) replace low variance
    info('Remove low variance features in data', ind=2)
    if isClassification(problemType):
        info('Classification is To do!', ind=4)
    if isRegression(problemType):
        info('Not removing any features since it is regression', ind=1)
        
        
    # 8) replace NA (if any remain)
    info('Replace NA (if any) in data', ind=2)
    replaceTargetNA(pddf, targetcol, targetNAstrategy)
    replaceFeatureNA(pddf, targetcol, featureNAstrategy)
    if sum(pddf.isnull().any()) > 0:
        error("There are still NA entries in the dataset!", ind=4)
        
    info('Finished formatting data. Data is now '+getDim(pddf), ind=2)
    
    return pddf