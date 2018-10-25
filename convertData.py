#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:59:24 2017

@author: tgadfort
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from colInfo import getDim, getNcols

from logger import info


def convertToBinaryInt(pddata, colname, positiveTarget):
    info('Convert Column [{0}] Data to Binary Integer'.format(colname), ind=4)
    
    uniqueValues = list(pddata[colname].unique())
    try:
        uniqueValues.index(positiveTarget)
    except:
        raise ValueError("Positive target",positiveTarget,"is not find in column data.")
                
    posLoc = pddata[colname] == positiveTarget
    negLoc = pddata[colname] != positiveTarget
    
    pddata[colname][posLoc] = 1
    pddata[colname][negLoc] = 0
    
    info('Set column [{0}] data to 1,0 for positive/negative classification'.format(colname), ind=4)

    

def labelEncodeData(colData):
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(colData)
    return colData.name, label_encoder, encoded



def dropEncodedColumns(pddata, columns):
    info('Dropping '+str(len(columns))+' encoded columns', ind=6)
    pddata.drop(columns, axis=1, inplace=True)
    
    
    
def getLabelEncoders(pddata):
    info('Label encoding data', ind=6)
    
    ## Find object columns
    encodedColumns = pddata.select_dtypes(include=['object']).columns
    info('Found '+str(len(encodedColumns))+' columns to encode', ind=6)


    ## create label encoders
    n_jobs=3
    results       = Parallel(n_jobs=n_jobs)(delayed(labelEncodeData)(pddata[cat_colname]) for cat_colname in encodedColumns)
    labelEncoders = {cat_colname: label_encoder for cat_colname, label_encoder, encoded in results}
    
    return labelEncoders, results



def getEncodedData(pddata):
    info('Convert Categorical Data To Integer', ind=4)


    ## label encode data
    labelEncoders, results = getLabelEncoders(pddata)


    ## create data frame of categorical features
    encodedCatData = pd.DataFrame({cat_colname: encoded for cat_colname, label_encoder, encoded in results})


    ## drop columns
    info('Dropping original '+getNcols(encodedCatData, asStr=True)+' columns', ind=6)
    dropEncodedColumns(pddata, encodedCatData.columns)
    info('Original data is now '+getDim(pddata), ind=6)


    ## join to original data
    #info('Joining encoded data', ind=6)
    #encodedData = encodedCatData.join(pddata)

    
    return pddata, encodedCatData, labelEncoders



def getHotEncodedData(encodedCatData, labelEncoders):
    info("One Hot Encoding Data", ind=4)
    if len(labelEncoders) == 0:
        return encodedCatData
    
    one_hot_encoder = OneHotEncoder()
    hotEncodedData  = one_hot_encoder.fit_transform(encodedCatData)
    encodedData     = pd.DataFrame(hotEncodedData.toarray())

    cat_colnames = labelEncoders.keys()
    n_values = [len(labelEncoders[cat_colname].classes_) for cat_colname in cat_colnames]
    colnames_with_dummy = []
    for column_cat, n_value in zip(cat_colnames, n_values):
        colnames_with_dummy += ['{0}:{1}'.format(column_cat, i) for i in range(n_value)]
    
    encodedData.columns = colnames_with_dummy
    
    return encodedData



def createInteractionTerms(Xdata):
    info("Creating interaction terms.", ind=4)
    Xdata = PolynomialFeatures(interaction_only=True).fit_transform(Xdata)



def scaleData(Xdata):
    info("Scaling data between [0,1]", ind=4)
    scaler = StandardScaler()
    scaler.fit(Xdata)

