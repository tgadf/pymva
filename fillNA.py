#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:49:58 2017

@author: tgadfort
"""

from sklearn.preprocessing import Imputer

from logger import info, error
from colInfo import getNcols


def replaceTargetNA(pddata, colname, strategy = 'zero'):
    info("Replace NA values for target ["+colname+"]", ind=4)
    if strategy == 'zero':
        pddata[colname].fillna(0, inplace=True)
    if strategy == 'mean':
        pddata[colname].fillna(pddata[colname].mean(), inplace=True)



def replaceFeatureNA(pddata, targetcol, strategy):
    info("Replace NA values for features", ind=4)

    nadata = pddata.loc[:, pddata.isnull().any()]
    info("Found "+getNcols(nadata, asStr=True)+" columns with an NA", ind=6)
    
    for dtype,dstrategy in strategy.iteritems():
        if dstrategy == None:
            continue
        columns = list(nadata.select_dtypes(include=[dtype]).columns)
        info("Replacing NA values for "+str(len(columns))+" "+dtype+" features", ind=6)
        if dstrategy == 'zero':
            for colname in columns:
                pddata[colname].fillna(0, inplace=True)
        if dstrategy == 'mean':
            for colname in columns:
                pddata[colname].fillna(pddata[colname].mean(), inplace=True)
        if dstrategy == 'dummy':
            for colname in columns:
                pddata[colname].fillna('dummy', inplace=True)
        if dstrategy == 'targetmean':
            for colname in columns:
                pddata.groupby(colname)[targetcol].transform('mean')
        
    nadata = pddata.loc[:, pddata.isnull().any()]
    ncols  = getNcols(nadata)
    if ncols > 0:
        info("Replaced all NAs", ind=4)
    else:
        info("There are still "+str(ncols)+" columns with an NA", ind=4)
    

def replaceNA(pddata, targetcol = None, impute = True, strategy='mean'):
    info("Replacing NA values in dataset.",ind=2)

    if impute:
        # The imputation strategy.
        # If “mean”, then replace missing values using the mean along the axis.
        # If “median”, then replace missing values using the median along the axis.
        # If “most_frequent”, then replace missing using the most frequent value along the axis.
    
        imputer = Imputer(strategy=strategy)
        if targetcol:
            pddata = imputer.fit_transform(X=pddata, y=pddata[targetcol])
        else:
            pddata = imputer.fit_transform(X=pddata)

    else:
        if strategy == 'zero':
            pddata['amount'].fillna(0, inplace=True)
        #if strategy == 'mean':
        #    pddata['amount'].fillna(pd, inplace=True)
            
    ## Test results        
    numNA = sum(pddata.isnull().sum())
    if numNA != 0:
        error("There are still NA values in dataset after Imputer.", ind=0, finish=True)

