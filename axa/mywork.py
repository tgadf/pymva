#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:05:07 2018

@author: tgadfort
"""


import sys
if '/Users/tgadfort/Python' not in sys.path:
    sys.path.insert(0, '/Users/tgadfort/Python')
if '/Users/tgadfort/Documents/pymva' not in sys.path:
    sys.path.insert(1, '/Users/tgadfort/Documents/pymva')

from numpy import append
from fileio import readCSVtoPandas, saveJoblib, getJoblib
from fsio import isFile, setDir, setFile
from fileinfo import getBasename
from search import findAll, findDirs
from visualization import histogram2D as hist2D

def readDriverTrips(driverID):
    basedir="/Users/tgadfort/Documents/pymva/axa/Axa-Insurance-Telematics-Kaggle"
    tripFiles = findAll(setDir(basedir, str(driverID)))
    data = []
    for i,tripFile in enumerate(tripFiles):
        tripData = readCSVtoPandas(tripFile, debug = False)
        data.append(tripData)

    print "Found {0} trips for driver {1}".format(len(data), driverID)        
    return data


def readTrips():
    drivers = findDirs("/Users/tgadfort/Documents/pymva/axa/Axa-Insurance-Telematics-Kaggle")
    drivers = [getBasename(x) for x in drivers]
    
    data = {}
    for driverID in drivers:
        print "Reading trips from driver {0}".format(driverID)
        data[driverID] = readDriverTrips(driverID)

    savefile = setFile("/Users/tgadfort/Documents/pymva/axa", "driverData.p")
    saveJoblib(savefile, data, compress=True)
    

def getTrips():
    savefile = setFile("/Users/tgadfort/Documents/pymva/axa", "driverData.p")
    data = getJoblib(savefile)
    return data


def combineXY(data):
    x = None
    y = None
    for driverID,driverData in data.iteritems():
        print "Getting {0} driver data.".format(driverID)
        for i,trip in enumerate(driverData):
            #print "\tGetting {0} trip data.".format(i)
            if x is not None:
                x = append(x, trip.x)
            else:
                x = trip.x

            if y is not None:
                y = append(y, trip.y)
            else:
                y = trip.y

        print "\t{0}".format(len(x))
        if len(x) > 1e6:
            break

    return x,y