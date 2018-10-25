#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 06:29:56 2018

@author: tgadfort
"""

from trippath import RoutePath
from driver import DriverModel

from fileio import saveJoblib
from fsio import setFile
from numpy import amax, amin, percentile, arange, zeros, ones, c_, nan_to_num, vstack
from timing import start, inter, end
from trip import tripFeatures

def createRouteFeatures(tripData, driverID, tripID):
    # initialize this path
    path = RoutePath(driverID,tripID)   # start with driver 1, route 1
       
    path.route = tripData.values
    path.time  = len(path.route)   # 1 second per data file


    # only analyze this path if it is not within a 90 meter bound of the starting point
    max_value = amax(path.route)
    min_value = amin(path.route)
    if ( max_value < 90 and min_value > -90):
        path.is_zero = 1   # this is a zero length route
        path.matched = 0   # the jitter is done differently
       

    # find the total distance along the route
    path.distance = path.get_route_distance(0, path.time)   
       
    path.energy_per_distance = path.total_energy / path.distance
    path.energy_per_time = path.total_energy / path.time
       
    path.setQuantiles()
    
    path.setSpeedTimes()
    
    path.simplify()
    
    features = path.getFeatures()
    
    return features
    

    
def createRoutes(driverData, driverID):
        
    list_of_paths = []
    for tripID,tripData in enumerate(driverData):
        path = createRouteFeatures(tripData, driverID, tripID)
        list_of_paths.append(path)
        #if tripID % 10 == 0: inter(t0, tripID, len(driverData))
            
    return list_of_paths


def generateRouteFeatures(data):
    t0 = start()
    features = {}
    for i,driverID in enumerate(data.keys()):
        driverData = data[driverID]
        print "Process driver {0}".format(driverID)
        features[driverID] = createRoutes(driverData, driverID)
        if i % 5 == 0: inter(t0, i, len(data))
    end(t0)    
    
    savefile = setFile("/Users/tgadfort/Documents/pymva/axa", "driverPaths.p")
    saveJoblib(savefile, features, compress=True)
    
    
    
def load_driver_trips(driver_id, driverData):
    dm = DriverModel(driver_id, driverData)
    simtraj, npst = dm.detectSimilarTraj()
    sim_groups = dm.computeSimilarTrips()
    driver_trips_id = arange(200)+1 + int(driver_id)*10000
    trips_id = npst[:,0] +1
    sim_trips = zeros(200)
    sim_trips_group = zeros(200)
    for group in sim_groups:
            gl = len(group)
            for gid in group: 
                    sim_trips[gid] = 1
                    sim_trips_group[gid] = gl
    res = c_[ones(200) * int(driver_id), driver_trips_id, trips_id, sim_trips, sim_trips_group]
    return res
    

def compute_aggmat_drivers(data):
    drivers = data.keys()
    numtrips = len(drivers) * 200
    
    print drivers

    driverID = drivers[0]
    driverData = data[driverID]
    dm = DriverModel(driverID, driverData)
    sizemat = len(dm.agg_headers)
    res = zeros((numtrips, sizemat), dtype='float')

    counter = 0

    for i,driverID in enumerate(drivers): 
        driverData = data[driverID]
        dm  = DriverModel(driverID, driverData)
        res = dm.agg_mat, dm.agg_headers
        #res = load_driver_aggmat_celery(i)
        #task_ids.append(res)
        
        result = load_driver_trips(driverID, data[drivers[i]])
        res[arange(counter*200,(counter+1)*200),:] = result.result        
    


    # RE ORDER
    ss = res[:,0].argsort()
    res = res[ss,:]
    
    return res


def generateDriverModels(data):
    t0 = start()
    features = {}
    for i,driverID in enumerate(data.keys()):
        driverData = data[driverID]
        print "Process driver {0}".format(driverID)
        dm  = DriverModel(driverID, driverData)
        results, headers = dm.agg_mat, dm.agg_headers    
        results = nan_to_num(results)
        features[driverID] = results
        if i % 5 == 0: inter(t0, i, len(data))
    end(t0)    
    
    savefile = setFile("/Users/tgadfort/Documents/pymva/axa", "driverModels.p")
    saveJoblib(savefile, features, compress=True)
    


def generateTripFeatures(data):
    t0 = start()
    features = {}
    for i,driverID in enumerate(data.keys()):
        driverData = data[driverID]
        print "Process driver {0}".format(driverID)
        results = None
        for j,trip in enumerate(driverData):
            tripResults = tripFeatures(trip.values)
            if results is None:
                results = tripResults
            else:
                results = vstack((results, tripResults))
            
        results = nan_to_num(results)
        features[driverID] = results
        if i % 5 == 0: inter(t0, i, len(data))
    end(t0)    
    
    savefile = setFile("/Users/tgadfort/Documents/pymva/axa", "driverTripFeatures.p")
    saveJoblib(savefile, features, compress=True)


