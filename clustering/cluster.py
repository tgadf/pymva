#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:44:20 2018

@author: tgadfort
"""

from sklearn.neighbors import NearestNeighbors


###############################################################################
#
# Nearest Neighbors
#
#   Penalty: None
#
###############################################################################
def createNearestNeighbors(params):
    # params['n_neighbors']
    # params['radius'] = 1
    # params[algorithm'] = {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    # params['leaf_size'] = 30
    # params['metric'] = {'minkowski', ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]}
    cls = NearestNeighbors()
    return cls