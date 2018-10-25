#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:47:15 2018

@author: tgadfort
"""


from scipy.stats import randint as rint
from scipy.stats import uniform as rfloat
from scipy.stats._distn_infrastructure import rv_frozen

from numpy import linspace, power


###############################################################################
#
# getParams
#
###############################################################################
def getParams(paramName, parType, parRange, params, estParams):
    if params is None:
        params = estParams
        
    paramValue = params.get(paramName)
    if paramValue is None:
        try:
            paramValue = estParams[paramName]
        except:
            raise ValueError(paramName,"is not a valid parameter")
        
    if not isinstance(paramValue, parType):
        raise ValueError(paramName,"is not a ",parType)
        
    if parRange is not None:
        if parType == int or parType == float:
            if paramValue < parRange[0] or paramValue > parRange[1]:
                raise ValueError(paramName,"value of",paramValue," is not in the valid range between",parRange)
        else:
            if paramValue not in parRange:
                raise ValueError(paramName,"value of",paramValue," is not one of",parRange)
    
    return paramValue



###############################################################################
#
# Merge Params
#
###############################################################################
def mergeParams(estimator, params):
    estParams   = estimator.get_params()
    if params is None:
        params = estParams
    else:
        params = dict(estParams.items() + params.items())

    return params            


###############################################################################
#
# setParams
#
###############################################################################
def setParam(paramName, params, tuneParams, force=False):
    tunes    = tuneParams.get(paramName)
    paramVal = params.get(paramName)
    if paramName not in params.keys():
        raise ValueError("Parameter {0} not found in {1}".format(paramName, params.keys()))

    if force is True:
        return tunes
    else:
        return paramVal
        
    if tunes is not None:
        if all(tunes):
            if paramVal is None:
                return tunes

            if isinstance(tunes[0], (int,float)):
                if paramVal < tunes[0] or paramVal > tunes[-1]:
                    raise ValueError("Parameter {0} has value {1} and is out of range [{1},{2}]".format(paramName, paramVal, tunes[0], tunes[-1]))
            else:
                if paramVal not in tunes:
                    raise ValueError("Parameter {0} has value {1} and is not in list {1}".format(paramName, paramVal, tunes))
        else:
            if paramVal not in tunes:
                raise ValueError("Parameter {0} is not in list {1}".format(paramName, tunes))
            
    return paramVal
    


###############################################################################
#
# getDistributionLimits
#
###############################################################################
def getDistributionLimits(dist):
    if isinstance(dist, rv_frozen):
        limits = list(dist.args)
    elif isinstance(dist, list):
        limits = dist
    else:
        raise ValueError("Unknown distribution type:",type(dist))
        
    return limits



###############################################################################
#
# Convert Distributions
#
###############################################################################
def convertLimits(limits, form = None):
    if not isinstance(limits, list):
        raise ValueError("No way to convert limits because {0} is not a list.".format(limits))


    if not all(limits):
        return limits  ## return list because there is a None in the list

        
    if form is None:
        if len(limits) < 2:
            raise ValueError("List {0} does not have two or more elements".format(limits))
        if isinstance(limits[0], int):
            dist = rint(limits[0], limits[-1])
        elif isinstance(limits[0], float):
            dist = rfloat(limits[0], limits[-1])
        elif isinstance(limits[0], str):
            dist = limits
        else:
            raise ValueError("Not sure how to process list {0}.".format(limits))
    else:
        if form == "uniform":
            if len(limits) < 2:
                raise ValueError("List {0} does not have two or more elements".format(limits))
            if isinstance(limits[0], int):
                dist = rint(limits[0], limits[-1])
            elif isinstance(limits[0], float):
                dist = rfloat(limits[0], limits[-1])
        else:
            raise ValueError("Form {0} is not supported.".format(form))

    return dist


def convertDistribution(dist):            
    if dist is None:
        raise ValueError("Distribution is None")
        
    if isinstance(dist, (int,float)):
        limits = dist
    elif isinstance(dist, list):
        limits = dist
    elif isinstance(dist, tuple):
        limits = dist
    elif isinstance(dist, rv_frozen):
        limits = roundParam([dist.a,dist.b])
    else:
        raise ValueError("Unknown distribution type:",type(dist))
        
    return limits



def roundParam(vals):    
    sig = max(getPrecision(min(vals)),getPrecision(max(vals)))
    if isinstance(vals, list):
        vals = [round(x,sig) for x in vals]
        
    return vals



def genLinear(minval, maxval, num = None, step = None):
    if step is not None and num is None:
        num = int((maxval - minval)/(step)) + 1
    elif step is None and num is not None:
        num = num
    else:
        raise ValueError("Need either step or num in genLinear")
    vals = list(linspace(minval, maxval, num))
    if isinstance(minval, int):
        vals = [int(x) for x in vals]
        return vals
    else:
        return roundParam(vals)


def genPowerTen(minval, maxval, num = None, step = None):
    vals = list(power(10, genLinear(float(minval), float(maxval), num, step)))
    return roundParam(vals)


def genPowerTwo(minval, maxval, num = None, step = None):
    vals = list(power(2, genLinear(float(minval), float(maxval), num, step)))
    return roundParam(vals)


def getPrecision(val):
    aval = abs(val)
    if isinstance(val, int):
        p = 1
    else:
        p = 1
    if aval > 1:
        while aval > 1 and aval > 0:
            aval /= 10
            p += 1
    else:
        while aval < 1 and aval > 0:
            aval *= 10
            p += 1

    return p