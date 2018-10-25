#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:45:01 2018

@author: tgadfort
"""


import numpy as np
import scipy.stats as ss
from math import cos, sin

def tripFeatures(trip,plotting=False):
    """
    Extracts features of a trip dataframe.
    OUTPUT:
        np.array including features
        list of angles between points in deg
    """
    
    # 1. duration
    duration = len(trip)    
    
    # 2. speed: euclidean distance between adjacent points
    speed = np.sum(np.diff(trip,axis=0)**2,axis=1)**0.5
    
    ### 2.1. smooth GPS data (by convolution) ####    
    smooth_speed =  movingaverage(speed,10) 
    #smooth_speed[np.where(smooth_speed>65)[0]] = smooth_speed[np.where(smooth_speed>65)[0]-1]
    
    # head changes
    head = np.diff(trip,axis=0)
    head_x,head_y = head[:,0],head[:,1]
    
    head_quantiles_x = ss.mstats.mquantiles(head_x,np.linspace(0.02,0.99,10))
    head_quantiles_y = ss.mstats.mquantiles(head_y,np.linspace(0.02,0.99,10))
    
    # compute speed statistics    
    #mean_speed = smooth_speed.mean()
    #max_speed = max(smooth_speed)
    std_speed = speed.std()
    # 3. acceleration
    smooth_accel = np.diff(smooth_speed)    
    
    # 3.1 get all negative acceleration values
    accel_s = np.array(smooth_accel)    
    neg_accel = accel_s[accel_s<0]
    pos_accel = accel_s[accel_s>0]

    # 3.3 average breaking strength
    #mean_breaking = neg_accel.mean()
    #mean_acceleration = pos_accel.mean() 
    
    # summary statistics
    std_breaking = neg_accel.std()
    std_acceleration = pos_accel.std()
    
    # 4. total distance traveled    
    total_dist = np.sum(smooth_speed,axis=0)               
    
    # 5. relative standzeit (last 5% are discarded due standing)
    last = int(round(len(trip)*0.05))
    eps = 1 # threshold for determining standing
    
    # relative standzeit
    speed_red = np.array(speed)[:last]
    standzeit = len(speed_red[speed_red<0+eps])/float(duration) 
    
    #### DRIVING STYLE REALTED FEATURES ####
    # 1. acceleration from stop
    
    # 1.1 get end of stops: where is speed near zero
    end_stops = stops(smooth_speed)        
    n_stops = len(end_stops) # how many stops
    
    # 1.2 how does the driver accelerate from stop?
    
    end_stops = end_stops.astype(int)[:-1,1]
    
    # following interval
    interval = 7 # 7 seconds following end of stop
    
    # only those which dont exceed indices of trip
    end_stops = end_stops[end_stops+interval<len(smooth_speed)-1]    
    n_stops = len(end_stops) 
    
    if n_stops>1:
        anfahren = np.zeros(shape=(1,n_stops)) # initialize array
        
        for i in range(n_stops):
            
            # slope at acceleration    
            start = end_stops[i]
            
            anfahren[0,i] =  np.diff([smooth_speed[start],smooth_speed[start+interval]])
           
    else:
        anfahren = np.array([0])
    # compute statistics
    mean_anfahren = anfahren.mean()
    max_anfahren = anfahren.max()
    std_anfahren = anfahren.std()
    
    # end cell
    last_cell = rounddown(normalize(trip[-2:,:]),30)[-1]
    
    # determine trip is a back-home trip
    if last_cell[0]==0 and last_cell[1]==0:
        hometrip=1
    else:
        hometrip=0
    
    # speed quantiles
    speed_quantiles = ss.mstats.mquantiles(smooth_speed,np.linspace(0.02,0.99,25)) 
    # acceleration quantiles
    accel_quantiles = ss.mstats.mquantiles(smooth_accel,np.linspace(0.02,0.99,25))
    
        
    #legend()
    ######################################    
    return np.concatenate((speed_quantiles,accel_quantiles,head_quantiles_x,head_quantiles_y,np.array([duration,total_dist,standzeit,std_speed,std_breaking,std_acceleration,std_anfahren,mean_anfahren,max_anfahren,n_stops,hometrip])))




# moving average smoothin filter
def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    
    smoothed = np.convolve(values, weigths, 'valid')
    return smoothed 
    
### 2. GEOMETRIC NORMALIZATION ###

def rotate(vector,angle, origin=(0, 0)):
    cos_theta, sin_theta = cos(angle), sin(angle)
    x0, y0 = origin

    vector = np.array(vector) -np.array(origin) 
    
    rotation_mat = np.array([[cos_theta,-sin_theta],[sin_theta,cos_theta]])
    #print rotation_mat
    
    vector_rotated =  np.dot(rotation_mat,vector)
    
    return vector_rotated
    
def flip(trip,horizontal=True,vertical=False):
    """
    Flips GPS coordinates of a trip along specified axis
    (horizontal or vertical).
    """
    
    if horizontal:
        trip[:,0]= -trip[:,0]
     
    return trip

def point_north(trip):
    """
    Rotates all GPS such that trip (vector of end point) points north.
    """
    
    # 1. determine angle between start-end vector and north pointing vector
    #north = [1,0]
    end = trip[-1,:] # "mean vector"
    
    angle = np.arctan2(end[0],end[1]) #angle_between(end,north) # in radians
    
    # 2. rotate all points
    rotated_trip = np.zeros(shape=(len(trip),2))
    
    for i in range(len(trip)):
        rotated_trip[i,:] = rotate(trip[i,:],angle)
        
    return(rotated_trip)

def normalize(trip):
        
    normed = point_north(trip)
    
    if normed[:,0].mean()<0:
        normed = flip(normed) 
    
        
    return(normed)
    
def rounddown(x,cellsize=10):
    ten =  np.floor(x / 10) * 10
    return np.trunc(ten/cellsize)
    
def stops(bits):   

  # make sure all runs of ones are well-bounded
  bounded = np.hstack(([1], bits, [1]))

  log = (bounded<0+0.5)*1
    
  # get 1 at run starts and -1 at run ends
  diffs = np.diff(log)    
  
  # get indices if starts and ends
  run_starts = np.where(diffs > 0)[0]
  run_ends = np.where(diffs < 0)[0]
  
  return np.array([run_starts,run_ends,run_ends-run_starts]).T
