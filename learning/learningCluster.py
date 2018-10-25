#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:14:56 2018

@author: tgadfort
"""

from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN, Birch


###############################################################################
#
# KMeans Clustering
#
#   Penalty: None
#
###############################################################################
def createKMeansClustering(params):
    # params['n_clusters'] = N
    # params['algorithm'] = {“auto”, “full” or “elkan”}
    cls = KMeans()
    return cls


###############################################################################
#
# MiniBatchKMeans Clustering
#
#   Penalty: None
#
###############################################################################
def createMiniBatchKMeansClustering(params):
    # params['n_clusters'] = N
    # params['algorithm'] = {“auto”, “full” or “elkan”}
    cls = MiniBatchKMeans()
    return cls


###############################################################################
#
# AffinityPropagation Clustering
#
#   Penalty: None
#
###############################################################################
def createAffinityPropagationClustering(params):
    # params['damping'] = [0.5, 1]
    # params['affinity'] = {“precomputed", "euclidean"}
    cls = AffinityPropagation()
    return cls


###############################################################################
#
# MeanShift Clustering
#
#   Penalty: None
#
###############################################################################
def createMeanShiftClustering(params):
    cls = MeanShift()
    return cls


###############################################################################
#
# Spectral Clustering
#
#   Penalty: None
#
###############################################################################
def createSpectralClustering(params):
    # params['n_clusters'] = [0.5, 1]
    # params['eigen_solver'] : {None, ‘arpack’, ‘lobpcg’, or ‘amg’}
    cls = SpectralClustering()
    return cls


###############################################################################
#
# Spectral Clustering
#
#   Penalty: None
#
###############################################################################
def createAgglomerativeClustering(params):
    # params['n_clusters'] = N
    # params['affinity'] = {“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or ‘precomputed’
    # params['linkage'] {“ward”, “complete”, “average”}
    cls = AgglomerativeClustering()
    return cls


###############################################################################
#
# Agglomerative Clustering
#
#   Penalty: None
#
###############################################################################
def createDBSCANClustering(params):
    # params['eps'] = 0.5
    # params['algorithm'] {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    cls = DBSCAN()
    return cls


###############################################################################
#
# Birch Clustering
#
#   Penalty: None
#
###############################################################################
def createBirchClustering(params):
    # params['n_clusters'] = N
    # params['threshold'] = 0.5
    cls = Birch()
    return cls