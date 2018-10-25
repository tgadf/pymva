#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:42:30 2018

@author: tgadfort
"""

from sklearn.model_selection import train_test_split

from logger import info, error
from colInfo import isColumn, getDim
from fsio import setFile, isFile
from fileio import getJoblib, saveJoblib
from mvapath import getDataDir

def getTrainTestNames(config):
    dname          = getDataDir(config)
    X_trainName    = setFile(dname, "X_train.p")
    X_testName     = setFile(dname, "X_test.p")
    X_validName    = setFile(dname, "X_valid.p")
    y_trainName    = setFile(dname, "y_train.p")
    y_testName     = setFile(dname, "y_test.p")
    y_validName    = setFile(dname, "y_valid.p")
    return X_trainName, X_testName, X_validName, y_trainName, y_testName, y_validName


def isSplitDataReady(config):
    X_trainName, X_testName, X_validName, y_trainName, y_testName, y_validName = getTrainTestNames(config)
    if all([isFile(X_trainName),isFile(X_testName),isFile(X_validName),
            isFile(y_trainName),isFile(y_testName),isFile(y_validName)]):
        return True
    else:
        return False
    
    
def getTrainData(config):
    X_trainName, X_testName, X_validName, y_trainName, y_testName, y_validName = getTrainTestNames(config)
    if isFile(X_trainName) and isFile(y_trainName):
        info("Loading {0}".format(X_trainName), ind=4)
        X_train  = getJoblib(X_trainName)
        info("Found data that is {0}".format(getDim(X_train)), ind=4)
        
        info("Loading {0}".format(y_trainName), ind=4)
        y_train  = getJoblib(y_trainName)
        info("Found data that is {0}".format(getDim(y_train)), ind=4)
        return X_train,y_train
    else:
        error("Train data is not ready")
        return None
    
    
def getTruthData(config):
    X_trainName, X_testName, X_validName, y_trainName, y_testName, y_validName = getTrainTestNames(config)
    if isFile(y_testName):
        info("Loading {0}".format(y_testName), ind=4)
        y_test  = getJoblib(y_testName)
        info("Found data that is {0}".format(getDim(y_test)), ind=4)
        return y_test
    else:
        error("Truth data is not ready")
        return None
        
    
def loadTrainTestData(config):
    X_trainName, X_testName, X_validName, y_trainName, y_testName, y_validName = getTrainTestNames(config)
    if all([isFile(X_trainName),isFile(X_testName),isFile(X_validName),
            isFile(y_trainName),isFile(y_testName),isFile(y_validName)]):
        info("Loading saved final train/test datasets.", ind=2)
        
        info("Loading {0}".format(X_trainName), ind=4)
        X_train = getJoblib(X_trainName)
        info("Found data that is {0}".format(getDim(X_train)), ind=4)        
        info("Loading {0}".format(X_testName), ind=4)
        X_test  = getJoblib(X_testName)
        info("Found data that is {0}".format(getDim(X_test)), ind=4)
        info("Loading {0}".format(X_validName), ind=4)
        X_valid = getJoblib(X_validName)
        info("Found data that is {0}".format(getDim(X_valid)), ind=4)
        
        info("Loading {0}".format(y_trainName), ind=4)
        y_train = getJoblib(y_trainName)
        info("Found data that is {0}".format(getDim(y_train)), ind=4)        
        info("Loading {0}".format(y_testName), ind=4)
        y_test  = getJoblib(y_testName)
        info("Found data that is {0}".format(getDim(y_test)), ind=4)
        info("Loading {0}".format(y_validName), ind=4)
        y_valid = getJoblib(y_validName)
        info("Found data that is {0}".format(getDim(y_valid)), ind=4)        

        return X_train, X_test, X_valid, y_train, y_test, y_valid
    else:
        error("Train/test datasets are not ready!")
        
    
def getTrainTestData(pddf, config):
    info("Creating final train/test datasets.", ind=0)
    
    ## Config info
    targetConfig   = config['target']
    targetcol      = targetConfig['colname']
    outputConfig   = config['output']
    compress       = outputConfig['compress']


    if not isColumn(pddf, targetcol):
        raise ValueError("Target column",targetcol,"is not included in data!")
    
    
    ## Determine if the data showed up split (seperate train/test files)
    isSplit = False
    isValid = False
    if isColumn(pddf, "isTrain"):
        info("Data is already split", ind=2)
        isSplit = True
    elif isColumn(pddf, "isValid"):
        info("Validation data is ready, but train/test data must be created", ind=2)
        isValid = True
    else:
        info("Train/test data must be created", ind=2)

    
    ## Create data if it's split
    if isSplit:
        info("Splitting train data", ind=2)
        X_train = pddf[pddf['isTrain'] == 1]
        y_train = X_train[targetcol]
        X_train.drop(labels=[targetcol, 'isTrain'], axis=1, inplace=True)
        
        info("Splitting test data", ind=2)
        X_test  = pddf[pddf['isTrain'] == 0]
        y_test  = X_test[targetcol]
        X_test.drop(labels=[targetcol, 'isTrain'], axis=1, inplace=True)
        
        X_valid = None
        y_valid = None
    elif isValid:
        info("Splitting validation data", ind=2)
        X_valid  = pddf[pddf['isValid'] == 1]
        y_valid  = X_valid[targetcol]
        
        info("Creating train/test data that contains validated data", ind=2)
        X_data  = pddf[pddf['isValid'] == 0]
        y = X_data[targetcol]
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)
    else:
        info("Creating train/test data that is not already split or validated", ind=2)
        y = pddf[targetcol]
        pddf.drop(labels=[targetcol], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(pddf, y, test_size=0.2)
        X_valid = None
        y_valid = None
        
    if isSplit:
        info("Dropping {0} from DataFrame".format(", ".join([targetcol, 'isTrain'])))
        pddf.drop(labels=[targetcol, 'isTrain'], axis=1, inplace=True)        
    elif isValid:
        info("Dropping {0} from DataFrame".format(", ".join([targetcol, 'isValid'])))
        pddf.drop(labels=[targetcol, 'isValid'], axis=1, inplace=True)        
        
    X_trainName, X_testName, X_validName, y_trainName, y_testName, y_validName = getTrainTestNames(config)


    info("Saving {0} data to {1}".format(getDim(X_train), X_trainName), ind=4)
    saveJoblib(X_trainName, X_train, compress)
    info("Saving {0} data to {1}".format(getDim(X_test), X_testName), ind=4)
    saveJoblib(X_testName, X_test, compress)
    info("Saving {0} data to {1}".format(getDim(X_valid), X_validName), ind=4)
    saveJoblib(X_validName, X_valid, compress)
        

    info("Saving {0} data to {1}".format(getDim(y_train), y_trainName), ind=4)
    saveJoblib(y_trainName, y_train, compress)
    info("Saving {0} data to {1}".format(getDim(y_test), y_testName), ind=4)
    saveJoblib(y_testName, y_test, compress)
    info("Saving {0} data to {1}".format(getDim(y_valid), y_validName), ind=4)
    saveJoblib(y_validName, y_valid, compress)


    
    return X_train, X_test, X_valid, y_train, y_test, y_valid

    