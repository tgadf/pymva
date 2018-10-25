#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:56:55 2018

@author: tgadfort
"""

from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA
from sklearn.decomposition import TruncatedSVD


###############################################################################
#
# PCA Decomposition
#
#   Penalty: None
#
###############################################################################
def createPCADecomposition(params):
    # params['svd_solver'] = {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
    cls = PCA()
    return cls



###############################################################################
#
# Incremental PCA Decomposition
#
#   Penalty: None
#
###############################################################################
def createIncrementalPCADecomposition(params):
    cls = IncrementalPCA()
    return cls



###############################################################################
#
# Kernel PCA Decomposition
#
#   Penalty: None
#
###############################################################################
def createKernelPCADecomposition(params):
    # params['kernel'] = {“linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”}
    # params['eigen_solver'] = [‘auto’|’dense’|’arpack’], default=’auto’

    # params['coef0'] = {1}
    # params['gamma'] = {1}
    # params['alpha'] = {1}

    cls = KernelPCA()
    return cls



###############################################################################
#
# Sparse PCA Decomposition
#
#   Penalty: None
#
###############################################################################
def createSparsePCADecomposition(params):
    # params['method'] = {‘lars’, ‘cd’}
    # params['alpha'] = {1}
    # params['ridge_alpha'] = {1}

    cls = SparsePCA()
    return cls



###############################################################################
#
# MiniBatch Sparse PCA Decomposition
#
#   Penalty: None
#
###############################################################################
def createMiniBatchSparsePCADecomposition(params):
    # params['method'] = {‘lars’, ‘cd’}
    # params['alpha'] = {1}
    # params['ridge_alpha'] = {1}

    cls = MiniBatchSparsePCA()
    return cls



###############################################################################
#
# MiniBatch Sparse PCA Decomposition
#
#   Penalty: None
#
###############################################################################
def createTruncatedSVDDecomposition(params):
    # params['n_components'] = 2

    cls = TruncatedSVD()
    return cls



#... More things in this class
