#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 06:37:32 2018

@author: tgadfort
"""


import numpy as np
#import matplotlib.pyplot as plt

from vector_math import get_distance, find_closest_point_at_minimum_distance
from vector_math import angle_between, match_RDP_to_route, rdp, get_angle_between_3_points
#import search_matches




class RoutePath:
    def __init__(self, driverid, routeid):
        self.driverid = driverid
        self.routeid  = routeid
        self.distance     = 0    # start with traveling zero distance
        self.time         = 0    # start with traveling for zero time
        
        self.feature_loc  = []
        self.angles       = []
          
        self.route        = []
        self.route_quantiles = []
        
        self.speed =    []
        self.speed_quantiles = []
        
        self.acceleration = []
        self.acceleration_quantiles = []
        
        self.total_energy = 0
        self.energy_per_distance = 0
        self.energy_per_time = 0
        
        self.time_in_speed = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0]
        self.time_at_speed = []

        self.angle_distances = []  # the distances between each of the angles
        
        self.comparison   = []
        
        self.matched      = -10   # default to not being matched
        
        self.is_zero      = 0  # if it is a zero distance route   
        
        self.print_flag   = 0
      
        
    def show(self):
        print "Driver ID: {0}".format(self.driverid)
        print "Route ID: {0}".format(self.routeid)
        print "Distance: {0}".format(self.distance)
        print "Time: {0}".format(self.time)
        print "Energy: {0}".format(self.total_energy)
        print "Route: {0}".format(self.route_quantiles)
        print "Speed: {0}".format(self.speed_quantiles)
        print "Acceleration: {0}".format(self.acceleration_quantiles)
        
        
    def simplify(self):
        self.distance        = round(self.distance, 2)
        self.total_energy    = round(self.total_energy, 2)
        self.route_quantiles = [round(x, 2) for x in self.route_quantiles]
        self.speed_quantiles = [round(x, 2) for x in self.speed_quantiles]
        self.acceleration_quantiles = [round(x, 2) for x in self.acceleration_quantiles]
        self.energy_per_distance = round(self.energy_per_distance,4)
        self.energy_per_time = round(self.energy_per_time,4)
      
      
    def setQuantiles(self):
        self.route_quantiles = np.percentile(self.route, q=[10, 30, 50, 70, 90])
        self.speed_quantiles = np.percentile(self.speed, q=[10, 30, 50, 70, 90])
        self.acceleration_quantiles = np.percentile(self.acceleration, q=[10, 30, 50, 70, 90])
        
                
    def setSpeedTimes(self):
        for i,timeInSpeed in enumerate(self.time_in_speed):
            timeAtSpeed = timeInSpeed / (self.time-2)
        self.time_at_speed.append(round(timeAtSpeed,5))
        
        self.energy_per_distance = self.total_energy / self.distance
        self.energy_per_time = self.total_energy / self.time
       

    def getFeatures(self):
        features = []
        features.append(self.driverid)
        features.append(self.routeid)
        features.append(self.distance)
        features.append(self.time)
        features.append(self.is_zero)
        features.append(self.speed_quantiles)
        features.append(self.acceleration_quantiles)
        features.append(self.total_energy)
        features.append(self.energy_per_distance)
        features.append(self.energy_per_time)
        features.append(self.time_at_speed)
        
        return features
    
         
    


    #********************
    #**** this function rotates the N x 2  path by a 2x2 rotation matrix
    #********************
    def rotate_path( self, angle_to_rotate):
   
      rotation_matrix = [   [ np.cos(angle_to_rotate), -1 * np.sin(angle_to_rotate) ], 
                            [ np.sin(angle_to_rotate),      np.cos(angle_to_rotate) ]  ]
      
      self.route = np.dot( self.route, rotation_matrix)
      return 
   
    
    #********************
    # This gets the distance traveled along a route   
    #********************
    def get_route_distance(self, start_id, end_id):
   
      total_distance = 0
      start_num = min(start_id, end_id)
      end_num = max(start_id, end_id)
      
      speed_range = [0,3,6,9,12,15,20,25,30,35,40,45,50,60]

      for cnt in range( start_num+2, end_num):
         x1 = self.route[ cnt-2, 0]
         y1 = self.route[ cnt-2, 1]

         x2 = self.route[ cnt, 0]
         y2 = self.route[ cnt, 1]   
   
         distance1 = get_distance(x1, y1, x2, y2)
         distance1 = distance1 / 2.0
         
         if (distance1 > 200):
            distance1 = 200
         
         self.speed.append(distance1)
         
         for cnt5 in range(0,len(speed_range)-2):
            if ( distance1 >= speed_range[cnt5] and distance1 < speed_range[cnt5+1]):
               self.time_in_speed[cnt5] += 1.0
         
         if (cnt > start_num+2):
            acceleration = self.speed[-1] - self.speed[-2]
            self.acceleration.append(acceleration)
            
            energy =  abs( self.speed[-1]**2 - self.speed[-2]**2)
            self.total_energy += energy
         
         total_distance += distance1
         
      return total_distance
      #********************
   # This gets the distance traveled along a route   
   #********************   
   
   
   #********************
   #**** end function rotates the N x 2  path by a 2x2 rotation matrix
   #********************






    #********************
    #**** this function finds the higest point and centers the grap on it
    #********************
    def center_on_highest_point(self):
      # find where the tallest point is at
      max_height = -1
      max_index = 0
      for cnt, coords in enumerate(self.route):
          if abs(coords[1] > max_height):
              max_height = abs(coords[1])
              max_index = cnt





      #if ( max_index > self.time/2.5):  # see if we are past halfway, if we are, flip in x
      #   self.flip_x_coords()

      max_index = max( 1,  min( max_index, self.time-2) )
      
      # center on that maximum value
      x_coord =  self.route[max_index,0]
      y_coord =  self.route[max_index,1]
      index_array = np.array([x_coord, y_coord])
      self.route = np.subtract(self.route, index_array)
      
      # see how far away the start and end points are
      dist_from_start = get_distance(x_coord, y_coord, self.route[0,0], self.route[0,1])
      dist_from_end = get_distance(x_coord, y_coord, self.route[-1,0], self.route[-1,1])
    
      # the farther away from the end points we are, the greater arm to center the angle on
      if (dist_from_start > 2500 and dist_from_end > 2500):  
         centering_distance = 1000
      elif (dist_from_start > 1500 and dist_from_end > 1500):
         centering_distance = 500
      elif (dist_from_start > 1000 and dist_from_end > 1000):
         centering_distance = 250
      elif (dist_from_start > 500 and dist_from_end > 500):
         centering_distance = 150
      elif (dist_from_start > 150 and dist_from_end > 150):
         centering_distance = 50
      else :
         centering_distance = 25
    
      loc_ahead  = find_closest_point_at_minimum_distance(self.route,max_index,centering_distance,1.0)
      loc_behind = find_closest_point_at_minimum_distance(self.route,max_index,centering_distance,-1.0)
      
      #print(max_index, loc_ahead, loc_behind)
      
      
      # get the angle between these vectors
      x0 =  self.route[loc_behind,0]
      y0 =  self.route[loc_behind,1]
      
      x1 =  self.route[max_index,0]
      y1 =  self.route[max_index,1]
      
      x2 =  self.route[loc_ahead,0]
      y2 =  self.route[loc_ahead,1]
      
      # get the vector
      v1 = [ x1-x0, y1-y0]
      v2 = [ x1-x2, y1-y2]
      v3 = [ 0, 1]
      
      
      angle1 = angle_between(v1, v2)   # the angle of the angle
      angle2 = angle_between(v3, v2)   # the angle of the vector ahead vs vertical
      
      #print("the angle is ",angle1, angle1 * 180 / np.pi)
      #print("the angle is ",angle2, angle2 * 180 / np.pi)
      
      target_angle = angle1 / 2.0  # we want the vertical to bisect our angle
      angle_diff =  -1 * (target_angle - angle2)
      
      self.rotate_path( angle_diff)  # rotate our path to bisect
       
      return 
   #********************
   #**** end function finds the higest point and centers the grap on it
   #********************




    #********************
    #**** this function flips the y coordinates
    #********************
    def flip_y_coords(self):
      flip_y = [1, -1]
      self.route = np.multiply( self.route, flip_y)
      return
   #********************
   #**** end function flips the y coordinates
   #********************

    #********************
    #**** this function flips the x coordinates
    #********************
    def flip_x_coords(self):
      flip_x = [-1, 1]
      self.route = np.multiply( self.route, flip_x)
      return
   #********************
   #**** end function flips the y coordinates
   #********************


    #**********************
    #*** this gets our list of features using an RDP sort
    #**********************
    def generate_features(self, rdp_tolerance):
   
      tolerance = rdp_tolerance
      simplified = np.array(rdp( self.route.tolist(), tolerance ))
      simplified_loc = match_RDP_to_route( simplified, self.route)
      features = []
      for cnt, item in enumerate(simplified_loc):
         features.append( [ simplified[cnt,0], simplified[cnt,1], item ] )
      self.feature_loc = np.array(features)

      return
   #**********************
   #*** this gets our list of features
   #**********************   


    #**********************
    #*** this gets our list of features using an RDP sort
    #**********************
    def update_feature_loc(self):
   
      current_loc = self.feature_loc[:,2]

      features = []
      for cnt, item in enumerate(current_loc):
         features.append( [ self.route[item,0], self.route[item,1], item ] )
      self.feature_loc = np.array(features)

      return
   #**********************
   #*** this gets our list of features
   #**********************      
   
   
   

    #**********************
    #*** this calculates angles and distances of the legs of the triangle that will be used for comparison
    #**********************
    #@profile
    def generate_angles(self):
   

      
      for cnt in  range(1, len(self.feature_loc) -1):  # get the angle between consecutive points
         x1 = self.feature_loc[cnt-1,0]
         y1 = self.feature_loc[cnt-1,1]

         x2 = self.feature_loc[cnt,0]
         y2 = self.feature_loc[cnt,1]
         
         x3 = self.feature_loc[cnt+1,0]
         y3 = self.feature_loc[cnt+1,1]         

         angle1, distance1, distance2 = get_angle_between_3_points(x1,y1,x2,y2,x3,y3)
  
         if (distance1> distance2):
            angle_info = [angle1, distance1,  distance2, cnt-1, cnt, cnt+1, len(self.angles)]
         else:
            angle_info = [angle1, distance2,  distance1, cnt+1, cnt, cnt-1, len(self.angles)]

         self.angles.append(angle_info)
         




      for cnt in  range(1, len(self.feature_loc) -2):  # get the angle between consecutive points, skipping 1
         x1 = self.feature_loc[cnt-1,0]
         y1 = self.feature_loc[cnt-1,1]

         x2 = self.feature_loc[cnt,0]
         y2 = self.feature_loc[cnt,1]
         
         x3 = self.feature_loc[cnt+2,0]
         y3 = self.feature_loc[cnt+2,1]         

         angle1, distance1, distance2 = get_angle_between_3_points(x1,y1,x2,y2,x3,y3)
  
         if (distance1> distance2):
            angle_info = [angle1, distance1,  distance2, cnt-1, cnt, cnt+2, len(self.angles)]
         else:
            angle_info = [angle1, distance2,  distance1, cnt+2, cnt, cnt-1, len(self.angles)]

         self.angles.append(angle_info)
         




      for cnt in  range(1, len(self.feature_loc) -2):  # get the angle between consecutive points, skipping 1
         x1 = self.feature_loc[cnt-1,0]
         y1 = self.feature_loc[cnt-1,1]

         x2 = self.feature_loc[cnt+1,0]
         y2 = self.feature_loc[cnt+1,1]
         
         x3 = self.feature_loc[cnt+2,0]
         y3 = self.feature_loc[cnt+2,1]         

         angle1, distance1, distance2 = get_angle_between_3_points(x1,y1,x2,y2,x3,y3)
  
         if (distance1> distance2):
            angle_info = [angle1, distance1,  distance2, cnt-1, cnt+1, cnt+2, len(self.angles)]
         else:
            angle_info = [angle1, distance2,  distance1, cnt+2, cnt+1, cnt-1, len(self.angles)]

         self.angles.append(angle_info)
         



      
      self.angles = np.array(self.angles)

      if (len(self.angles) > 0):
          self.angles = self.angles[ self.angles[:,0].argsort() ]  # sort from smallest angle to biggest angle
          
          for cnt in xrange(0,len(self.angles)):    # remember where this is in the array
             self.angles[cnt,6] = cnt     
      
      
      # get the distance between all of the angles
      self.angle_distances = np.zeros( (len(self.angles), len(self.angles)), dtype=np.float64 )  # start off at zero distance
      
      # use fortran to get the distance between the center point of each of the angles
      if (len(self.angles) > 0):
          #angles1 = self.angles[:,:]
          #features = self.feature_loc[:,:]
          raise ValueError("Not sure what to do here!")
          #self.angle_distances = search_matches.calculate_angle_distances(angles1, features)
      

      #for i in range (0, len(self.angles)-1):
      #    for j in range(i+1, len(self.angles)):
      #   
      #        cnt1 = self.angles[i,4]  # the the coordinates of the sorted angles
      #        cnt2 = self.angles[j,4]
      #        
      #        #print( cnt1, cnt2, angles1[i,4], angles1[j,4] )
      #        
      #    
      #        x1, y1 = self.feature_loc[cnt1,0], self.feature_loc[cnt1,1]  # the the coordinates of the sorted angles
      #        x2, y2 = self.feature_loc[cnt2,0], self.feature_loc[cnt2,1]
      #        distance1 = get_distance(x1,y1,x2,y2)
      #
      #        self.angle_distances[i,j] = distance1  # remember the distances between these angles
      #        self.angle_distances[j,i] = distance1  # remember the distances between these angles
      #        
      #        #if ( abs(test_angle_distances[i,j] - distance1)  > .001 or  abs(test_angle_distances[j,i] - distance1)  > .001):
      #        #   print i,j,test_angle_distances[i,j],distance1
      #        #   sys.exit(0)
              

      return   
   #**********************
   #*** this calculates angles and distances of the legs of the triangle that will be used for comparison
   #**********************   