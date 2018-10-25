#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:58:20 2017

@author: tgadfort
"""

from logger import info
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_string_dtype, is_numeric_dtype


def getColumnNames(pddata):
    return pddata.columns

def getColumnDataName(colData):
    return colData.name

def isColumn(pddata, colname):
    if colname:
        if colname in getColumnNames(pddata):
            return True
    return False


def getNrows(pddata, asStr = False):
    nrows = pddata.shape[0]
    if asStr:
        nrows = str(nrows)
    return nrows

def getNcols(pddata, asStr = False):
    ncols = pddata.shape[1]
    if asStr:
        ncols = str(ncols)
    return ncols


def getDim(pddata):
    if pddata is not None:
        return ' x '.join([str(x) for x in pddata.shape])
    else:
        return '0 x 0'


def getColumnMetaData(pddata, colname, infoname = "sum"):
    if isColumn(pddata, colname):
        if infoname == "sum":
            retval = pddata[colname].mean()
        elif infoname == "mean":
            retval = pddata[colname].mean()
        elif infoname == "count":
            retval = pddata[colname].count()
        elif infoname == "nNA":
            retval = sum(pddata[colname].isnull())
        elif infoname == "min":
            retval = pddata[colname].min()
        elif infoname == "max":
            retval = pddata[colname].max()
        elif infoname == "var":
            retval = pddata[colname].var()
        elif infoname == "std":
            retval = pddata[colname].std()
        elif infoname == "unique":
            retval = len(pddata[colname].unique())
        else:
            raise ValueError("Info",infoname,"is not available")
    else:
        raise ValueError("Column",colname,"does not exist in df")
        
    return retval


def isNumeric(colData):
    return is_numeric_dtype(colData)


def isInteger(colData):
    return is_numeric_dtype(colData)

def getColType(colData, testType = None):
    info('Checking column data type', ind=4)
    
    ctype = None
    if is_bool_dtype(colData):
        ctype = bool
    elif is_float_dtype(colData):
        ctype = float
    elif is_integer_dtype(colData):
        ctype = int
    elif is_string_dtype(colData):
        ctype = str
    else:
        raise ValueError("Not sure about dtype:",colData.dtype)

    if testType:
        if testType == ctype:
            return True
        else:
            return False
        
    return ctype