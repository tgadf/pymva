#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:31:30 2018

@author: tgadfort
"""

from downloadAll import load_data

# Feature Analysis Imports
# NOTE that all these are available for import directly from the `yellowbrick.features` module
from yellowbrick.features.rankd import Rank1D, Rank2D
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.pcoords import ParallelCoordinates
from yellowbrick.features.jointplot import JointPlotVisualizer
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.scatter import ScatterVisualizer

def showRadVis():
    # Load the classification data set
    data = load_data('occupancy')
    
    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()
    
    # Instantiate the visualizer
    visualizer = RadViz(classes=classes, features=features)
    
    visualizer.fit(X, y)      # Fit the data to the visualizer
    visualizer.transform(X)   # Transform the data
    visualizer.poof()         # Draw/show/poof the data
    
    
def showRank1D():
    # Load the dataset
    data = load_data('credit')
    
    # Specify the features of interest
    features = [
            'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
            'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
            'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
            'jul_pay', 'aug_pay', 'sep_pay',
        ]
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.default.as_matrix()
    
    # Instantiate the 1D visualizer with the Sharpiro ranking algorithm
    visualizer = Rank1D(features=features, algorithm='shapiro')

    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.poof()                   # Draw/show/poof the data
    
    
def showRank2D():
    # Load the dataset
    data = load_data('credit')
    
    # Specify the features of interest
    features = [
            'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
            'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
            'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
            'jul_pay', 'aug_pay', 'sep_pay',
        ]
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.default.as_matrix()
    
    # Instantiate the visualizer with the Covariance ranking algorithm
    visualizer = Rank2D(features=features, algorithm='pearson')
    
    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.poof()                   # Draw/show/poof the data
    
    
def showParallelCoordinates():
    # Load the classification data set
    data = load_data('occupancy')
    
    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()
    # Instantiate the visualizer
    visualizer = ParallelCoordinates(classes=classes, features=features)
    
    visualizer.fit(X, y)      # Fit the data to the visualizer
    visualizer.transform(X)   # Transform the data
    visualizer.poof()         # Draw/show/poof the data
    
    
    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
            classes=classes, features=features,
            normalize='standard', sample=0.1,
            )

    visualizer.fit(X, y)      # Fit the data to the visualizer
    visualizer.transform(X)   # Transform the data
    visualizer.poof()         # Draw/show/poof the data
    
    
def showPCAProjection():
    # Load the classification data set
    data = load_data('credit')
    
    # Specify the features of interest
    features = [
        'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
        'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
        'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
        'jul_pay', 'aug_pay', 'sep_pay',
    ]
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.default.as_matrix()
    visualizer = PCADecomposition(scale=True, center=False, col=y)
    visualizer.fit_transform(X,y)
    visualizer.poof()
    
    visualizer = PCADecomposition(scale=True, center=False, col=y, proj_dim=3)
    visualizer.fit_transform(X,y)
    visualizer.poof()
    
    
    
def showDirectDataVisualization():
    # Load the classification data set
    data = load_data('occupancy')
    
    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']
    
    # Extract the numpy arrays from the data frame
    X = data[features]
    y = data.occupancy
    
    visualizer = ScatterVisualizer(x='light', y='C02', classes=classes)
    
    visualizer.fit(X, y)
    visualizer.transform(X)
    visualizer.poof()
    
    
    
    # Load the data
    df = load_data('concrete')
    feature = 'cement'
    target = 'strength'
    
    # Get the X and y data from the DataFrame
    X = df[feature]
    y = df[target]
    visualizer = JointPlotVisualizer(feature=feature, target=target)
    
    visualizer.fit(X, y)
    visualizer.poof()
    
    visualizer = JointPlotVisualizer(
            feature=feature, target=target, joint_plot='hex'
            )

    visualizer.fit(X, y)
    visualizer.poof()