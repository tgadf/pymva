#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:37:19 2018

@author: tgadfort
"""

from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding, MDS, TSNE


###############################################################################
#
# Isomap Manifold
#
###############################################################################
def createIsomapManifoldLearning(params):
    # params['n_neighbors'] = N
    # params['eigen_solver'} = [‘auto’|’arpack’|’dense’]
    # params['path_method'] = {‘auto’|’FW’|’D’}
    # params['neighbors_algorithm'] = {‘auto’|’brute’|’kd_tree’|’ball_tree’}
    mfd = Isomap()
    return mfd


###############################################################################
#
# LocallyLinearEmbedding Manifold
#
###############################################################################
def createLocallyLinearEmbeddingLearning(params):
    # params['n_neighbors'] = N
    # params['eigen_solver'} = [‘auto’|’arpack’|’dense’]
    # params['method'] = (‘standard’, ‘hessian’, ‘modified’ or ‘ltsa’)
    # params['neighbors_algorithm'] = {‘auto’|’brute’|’kd_tree’|’ball_tree’}
    mfd = LocallyLinearEmbedding()
    return mfd


###############################################################################
#
# SpectralEmbedding Manifold
#
###############################################################################
def createSpectralEmbeddingLearning(params):
    # params['affinity'] = {‘nearest_neighbors’, ‘rbf’, ‘precomputed’}
    # params['eigen_solver'} = [‘auto’|’arpack’|’dense’]
    # params['n_neighbors'] = Optional
    mfd = SpectralEmbedding()
    return mfd


###############################################################################
#
# MDS Manifold
#
###############################################################################
def createMDSLearning(params):
    # params['dissimilarity'] : ‘euclidean’ | ‘precomputed’
    mfd = MDS()
    return mfd


###############################################################################
#
# TSNE Manifold
#
###############################################################################
def createTSNELearning(params):
    # params['learning_rate'] [10, 1000]
    mfd = MDS()
    return mfd