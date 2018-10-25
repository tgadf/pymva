#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:19:14 2017

@author: tgadfort
"""


import sys
if '/Users/tgadfort/Python' not in sys.path:
    sys.path.insert(0, '/Users/tgadfort/Python')
    
from pandas import read_csv, DataFrame, Series, to_numeric
from numpy import repeat, append
from sklearn import datasets

from logger import info
from mvapath import getDataDir
from fileio import getJoblib,saveJoblib
from fsio import setFile, isFile
from colInfo import getDim
from analyzeColumnData import writeDropList



def readCSV(filename):
    info("Reading data ["+filename+"]")
    pddata = read_csv(filename, low_memory=False)

    info("Read data with size "+getDim(pddata))
    return pddata


def readKeyValue(filename):
    info("Reading dict file ["+filename+"] into data frame", ind=6)
    pddata = DataFrame(filename)
    return pddata


def joinTrainTest(trainData, testData):
    info("Joining train and test data", ind=6)
    trainVal  = Series(repeat(1, trainData.shape[0]))
    trainData = trainData.assign(isTrain=trainVal.values)
    info("Train data has size "+getDim(trainData), ind=6)
    testVal   = Series(repeat(0, testData.shape[0]))
    testData  = testData.assign(isTrain=testVal.values)
    info("Test data has size "+getDim(testData), ind=6)
    pddf = trainData.append(testData)
    info("Combined data has size "+getDim(pddf), ind=6)        
    return pddf


def joinTrainValid(trainData, validData):
    info("Joining train and validation data", ind=6)
    validVal  = Series(repeat(0, trainData.shape[0]))
    trainData = trainData.assign(isValid=validVal.values)
    info("Train data has size "+getDim(trainData), ind=6)
    validVal   = Series(repeat(1, validData.shape[0]))
    validData  = validData.assign(isValid=validVal.values)
    info("Validation data has size "+getDim(validData), ind=6)
    pddf = trainData.append(validData)
    info("Combined data has size "+getDim(pddf), ind=6)        
    return pddf



def readDataset(config, name):
    info("Loading {0} dataset".format(name), ind=0)
    if name == "boston":
        ddata = datasets.load_boston()
    elif name == "diabetes":
        ddata = datasets.load_diabetes()
    elif name == "digits":
        ddata = datasets.load_diabetes()
    elif name == "cancer":
        ddata = datasets.load_breast_cancer()
    elif name == "wine":
        ddata = datasets.load_wine()
    else:
        raise ValueError("Name {0} is not recognized in readDatasets".format(name))
        
    X = ddata.data
    y = ddata.target
    y = y.reshape((y.shape[0],1))
    df = append(arr=X, values=y, axis=1)
    pddf = DataFrame(df)
    columns = list(ddata.feature_names)
    columns.append("TARGET")
    pddf.columns = columns
    return pddf



def makeDataset(config, name):
    info("Making {0} dataset".format(name))
    if name == "regression":
        X, y = datasets.make_regression(50000, 10, noise=10)
    elif name == "classification":
        X, y = datasets.make_classification(50000, 10)
    else:
        raise ValueError("Name {0} is not recognized in makeDataset".format(name))
        
    y = y.reshape((y.shape[0],1))
    df = append(arr=X, values=y, axis=1)
    pddf = DataFrame(df)
    columns = ["Var"+str(x) for x in range(X.shape[1])]
    columns.append("TARGET")
    pddf.columns = columns        
        
    return pddf


def makeRegression(config):
    return makeDataset(config, "regression")


def makeClassification(config):
    return makeDataset(config, "classification")



def readKDD99(config):
    info("Getting KDD '99 data.")
    
    datadir      = getDataDir(config)
    outputConfig = config['output']
    compress     = outputConfig['compress']
    dataName     = setFile(datadir, outputConfig['name'])
    
    featureConfig = config['feature']
    dlFile        = setFile(datadir, featureConfig['dropList'])
    
    
    if isFile(dataName) and isFile(dlFile):
        info("Loading previously create data frames")
        pddf = getJoblib(dataName)
    else:
        info("Downloading KDD '99 data",ind=2)
        tmp = datasets.fetch_kddcup99()
        X = tmp['data']
        y = tmp['target']
        y = y.reshape((y.shape[0],1))
        pddf = DataFrame(append(arr=X, values=y, axis=1))

        tmp = pddf.head(n=1000)
        for column in tmp.columns:
            try:
                tmp[column].mean()
                pddf[column] = to_numeric(pddf[column], errors="coerce")
            except:
                continue
        
        colFile  = setFile(datadir, "names.dat")
        colnames = open(colFile).readlines()
        targets  = colnames[0].split(",")
        columns  = [x.split(":")[0] for x in colnames[1:]]
        columns.append("TARGET")
        pddf.columns = columns

        info("Saving data to {0}".format(dataName))        
        saveJoblib(jlfile=dataName, jldata=pddf, compress=compress)
        
        info("Saving feature data to {0}".format(dlFile))
        writeDropList(dlFile, pddf, dlData = None)
        
    return pddf



def readUptake(config):
    info("Getting uptake data.")
    

    datadir      = getDataDir(config)
    outputConfig = config['output']
    compress     = outputConfig['compress']
    dataName     = setFile(datadir, outputConfig['name'])
    
    inputConfig  = config['input']
    trainName    = setFile(datadir, inputConfig['train'])
    validName    = setFile(datadir, inputConfig['valid'])
        
    featureConfig = config['feature']
    dlFile        = setFile(datadir, featureConfig['dropList'])
    
    
    if isFile(dataName) and isFile(dlFile):
        info("Loading previously create data frames")
        pddf = getJoblib(dataName)
    else:
        trainData = readCSV(trainName)
        validData = readCSV(validName)
        
        ## Remove 'market' from validation data
        validData.drop(labels='market', axis=1, inplace=True)
        
        pddf = joinTrainValid(trainData, validData)
        info("Saving training and validation data")
        saveJoblib(dataName, pddf, compress)
        info("Wrote training and validation data to "+dataName)
        
        info("Saving feature data")
        writeDropList(dlFile, pddf, dlData = None)
        
    return pddf

 
def readData(config):
    info("Getting data for analysis")

    ## Get name    
    name     = config['name']
    
    
    ## Load the data we need
    if name == "uptake":
        pddf = readUptake(config)
    elif name == "kdd99":
        pddf = readKDD99(config)
    elif name in ["boston", "diabetes", "wine", "digits", "cancer"]:
        pddf = readDataset(config, name)
    elif name == "regression":
        pddf = makeRegression(config)
    elif name == "classification":
        pddf = makeClassification(config)
    else:
        raise ValueError("Name",name,"not recognized in readData()")
        
    info("Using data that is "+getDim(pddf), ind=0)
    return pddf