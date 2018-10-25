#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:45:09 2018

@author: tgadfort
"""

from logger import info
from paramHelper import setParam, mergeParams, convertDistribution
from paramHelper import genLinear, genPowerTen

from pyearth import Earth

###############################################################################
#
# EARTH Regression
#
#  git clone git://github.com/scikit-learn-contrib/py-earth.git
#  The R package earth was most useful to me in understanding the algorithm, particularly because of Stephen Milborrow's thorough and easy to read vignette (http://www.milbo.org/doc/earth-notes.pdf).
#
###############################################################################
def createEARTHRegressor(params = None):
    info("Creating EARTH Regressor", ind=4)
    
    ## Params
    params     = mergeParams(Earth(), params)
    tuneParams = getEarthParams()


    info("Without Parameters", ind=4)
    
    
    # Estimator
    reg = Earth()
        
    return {"estimator": reg, "params": tuneParams}
    


def getEarthParams():
    retval = {"dist": {}, "grid": {}}
    return retval

#
#model.fit(X,y)
#    
#Print the model
#print(model.trace())
#print(model.summary())
#    
##Plot the model
#y_hat = model.predict(X)
#pyplot.figure()
#pyplot.plot(X[:,6],y,'r.')
#pyplot.plot(X[:,6],y_hat,'b.')
#pyplot.xlabel('x_6')
#pyplot.ylabel('y')
#pyplot.title('Simple Earth Example')
#pyplot.show()