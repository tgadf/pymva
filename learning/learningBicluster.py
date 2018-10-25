#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:49:18 2018

@author: tgadfort
"""

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster.bicluster import SpectralBiclustering

###############################################################################
#
# Spectral Coclustering
#
#   Penalty: None
#
###############################################################################
def createSpectralCoclustering(params):
    # params['n_clusters'] = N
    cls = SpectralCoclustering()
    return cls



###############################################################################
#
# Spectral Biclustering
#
#   Penalty: None
#
###############################################################################
def createSpectralBiclustering(params):
    # params['n_clusters'] = N
    # params['method'] = {‘scale’, ‘bistochastic’, or ‘log’]
    cls = SpectralBiclustering()
    return cls