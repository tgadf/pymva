#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:09:09 2018

@author: tgadfort
"""

from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import MiniBatchKMeans


def showElbow():
    # Make 8 blobs dataset
    X, y = make_blobs(centers=8)
    # Instantiate the clustering model and visualizer
    visualizer = KElbowVisualizer(MiniBatchKMeans(), k=(4,12))
    
    visualizer.fit(X) # Fit the training data to the visualizer
    visualizer.poof() # Draw/show/poof the data
    
    
def showSilhouette():
    # Make 8 blobs dataset
    X, y = make_blobs(centers=8)
    # Instantiate the clustering model and visualizer
    model = MiniBatchKMeans(6)
    visualizer = SilhouetteVisualizer(model)
    
    visualizer.fit(X) # Fit the training data to the visualizer
    visualizer.poof() # Draw/show/poof the data