#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:01:00 2018

@author: tgadfort
"""

from downloadAll import load_data

# Classifier Evaluation Imports

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance

def showReport():
    # Load the classification data set
    data = load_data('occupancy')
    
    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()
    
    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Instantiate the classification model and visualizer
    bayes = GaussianNB()
    visualizer = ClassificationReport(bayes, classes=classes)
    
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof()             # Draw/show/poof the data
    
    
def showConfusionMatrix():
    #First do our imports
    
    from sklearn.datasets import load_digits
    
    from yellowbrick.classifier import ConfusionMatrix
    # We'll use the handwritten digits data set from scikit-learn.
    # Each feature of this dataset is an 8x8 pixel image of a handwritten number.
    # Digits.data converts these 64 pixels into a single array of features
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=11)
    
    model = LogisticRegression()
    
    #The ConfusionMatrix visualizer taxes a model
    cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])
    
    #Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
    cm.fit(X_train, y_train)
    
    #To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
    #and then creates the confusion_matrix from scikit learn.
    cm.score(X_test, y_test)
    
    #How did we do?
    cm.poof()
    
    
def showROC():
    # Load the classification data set
    data = load_data('occupancy')
    
    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()
    
    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Instantiate the classification model and visualizer
    logistic = LogisticRegression()
    visualizer = ROCAUC(logistic)
    
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof()             # Draw/show/poof the data
    
    
def showClassBalance():
    # Load the classification data set
    data = load_data('occupancy')
    
    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']
    
    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()
    
    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Instantiate the classification model and visualizer
    forest = RandomForestClassifier()
    visualizer = ClassBalance(forest, classes=classes)
    
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof()             # Draw/show/poof the data