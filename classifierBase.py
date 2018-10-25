#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:39:33 2018

@author: tgadfort
"""

from logger import info, error
from numpy import linspace, power

class param:
    
    def __init__(self, paramName, paramData):
        #info("Param {0}: {1}".format(paramName, paramData), ind=6)
        self.name = paramName
        self.data = paramData
        self.vals = []
        
        if isinstance(paramData, (int, float)):
            self.vals = [paramData]
        elif isinstance(paramData, list):
            if isinstance(paramData[0], (int,float)) and isinstance(paramData[-1], (int,float)):
                if len(paramData) == 3:
                    try:
                        self.vals = self.genLinear(minval=paramData[0],
                                                   maxval=paramData[1],
                                                   num=paramData[2])
                    except:
                        self.vals = paramData
                else:
                    self.vals = paramData
            elif isinstance(paramData[0], (str,type(None))):
                self.vals = paramData
            else:
                raise ValueError("Could not construct param values from {0}".format(paramData))
        else:
            try:
                self.vals = eval('self.'+paramData)
            except:
                raise ValueError("Could not evaluate 'self.{0}'".format(paramData))

        info("Param {0}: {1}".format(paramName, self.vals), ind=6)
        #info("Values ---> {0}".format(self.vals), ind=8)

    def getName(self):
        return self.name

    def getVals(self):
        return self.vals

    def get(self):
        return {"name": self.name, "vals": self.vals}


    def roundParam(self, vals):    
        sig = max(self.getPrecision(min(vals)),self.getPrecision(max(vals)))
        if isinstance(vals, list):
            vals = [round(x,sig) for x in vals]            
        return vals

    def getPrecision(self, val):
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
    
    def genLinear(self, minval, maxval, num = None, step = None):
        if step is not None and num is None:
            num = int((maxval - minval)/(step)) + 1
        elif step is None and num is not None:
            step = (maxval-minval)/(num - 1)
        else:
            raise ValueError("Need either step or num in genLinear")
        vals = list(linspace(minval, maxval, num))
        if isinstance(step, int):
            vals = [int(x) for x in vals]
            return vals
        else:
            return self.roundParam(vals)

    
    def genPowerTen(self, minval, maxval, num = None, step = None):
        vals = list(power(10, self.genLinear(float(minval), float(maxval), num, step)))
        return self.roundParam(vals)

    
    def genPowerTwo(self, minval, maxval, num = None, step = None):
        vals = list(power(2, self.genLinear(float(minval), float(maxval), num, step)))
        return self.roundParam(vals)

    
    def genList(self, val):
        return val


        

class classifier:
    def __init__(self, name, estimator = None, params = None):
        info("Creating {0} Classifier".format(name), ind=4)
        self.name   = name
        self.estimator = estimator
        if estimator is not None:
            self.params = estimator.get_params()
        self.tunes  = []
        if isinstance(params, dict):
            for paramName,paramData in params.iteritems():
                paramObj = param(paramName, paramData)
                paramVal = paramObj.getVals()
                if len(paramVal) == 1 and self.estimator is not None:
                    self.setParam(paramName, paramVal[0])
                else:
                    self.tunes.append(paramObj)
        self.params = {}
        

    def addParam(self, paramName, paramData):
        paramObj = param(paramName, paramData)
        self.tunes.append(paramObj)
        
    def setParam(self, paramName, paramValue):
        test = 'self.estimator.{0}={1}'.format(paramName, paramValue)
        try:
            exec(test)
            info("Fixed Value ---> {0}".format(paramValue), ind=8)
        except:
            error("Could not set {0} to {1}".format(paramName, paramValue))
    
    def addEstimator(self, estimator):
        self.estimator = estimator
        self.params = estimator.get_params()
        
    def fit(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        
    def predict(self, X_test):
        self.estimator.predict(X_test)
        
    def predict_proba(self, X_test):
        self.estimator.predict_proba(X_test)

        
    def getEstimator(self):
        return self.estimator
        
    def getParams(self):
        return self.tunes
    
    
    def show(self):
        info("Classifier: {0}".format(self.name), ind=2)
        info("Estimator", ind=2)
        info("{0}".format(self.estimator))
        info("Tunable Parameters", ind=2)
        for tune in self.tunes:            
            info("Param: {0} --> [{1}]".format(tune.getName(), tune.getVals()), ind=4)
        info("Default Parameters", ind=2)
        for name,val in self.params.iteritems():            
            info("Param: {0} --> [{1}]".format(name, val), ind=4)

    def get(self):
        params = {"grid": {}, "dist": {}}
        for tune in self.tunes:
            param = tune.get()
            params['grid'][param['name']] = param['vals']
            params['dist'][param['name']] = param['vals']
        return {"estimator": self.estimator, "params": params}