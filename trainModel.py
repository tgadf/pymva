#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:30:59 2018

@author: tgadfort
"""

from logger import info, error
from colInfo import getDim
from targetInfo import isClassification, isRegression, isClustering
from models import getModel, getModelData
from mvapath import getModelsDir
from fileio import saveJoblib, getJoblib
from fsio import setFile

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#from numpy import where
#where(modelData is not None and modelData.get('tune') is True, False, True).tolist()


###########################################################################
#
# Train Model
#
###########################################################################
def trainModel(modelname, X_train, y_train, config):    
    info("Training a {0} estimator".format(modelname), ind=0)
    info("X data is {0}".format(getDim(X_train)), ind=2)
    info("y data is {0}".format(getDim(y_train)), ind=2)
    
    problemType = config['problem']
    info("This is a {0} problem".format(problemType), ind=2)
    
    modelData = getModelData(config, modelname)
    tuneForParams = True
    refitModel = False
    goodModel = True
    if modelData is not None:
        if modelData.get('tune') is False:
            tuneForParams = False
        if modelData.get('fit') is True:
            tuneForParams = False
        if modelData.get('cv') is True:
            tuneForParams = False
        if modelData.get('refit') is True:
            refitModel = True
        if modelData.get('error') is True:
            goodModel = False
    else:
        info("No model parameters were given. Using default {0} estimator".format(modelname), ind=4)
        tuneForParams = False

    if goodModel is False:
        error("Model {0} is no good and will not run it.".format(modelname))
        return None
    

    #################################################################
    # Get Model
    #################################################################
    retval = getModel(config, modelname)


    #################################################################
    # Tune Parameters
    #################################################################
    estimator = retval['estimator']
    params    = retval['params']
    

    if tuneForParams:
        tuneResults = tuneModel(modelname, estimator, params, X_train, y_train, config)
        estimator   = tuneResults['estimator']
        params      = tuneResults['params']
        
        if refitModel:
            try:
                estimator.set_params(probability=True)
                info("Set probability to True for model refit", ind=4)
            except:
                info("Could not set probability to True for model refit")
            info("Re-fitting for {0} model parameters with probability".format(modelname), ind=4)
            estimator = estimator.fit(X_train, y_train)
            info("Finished re-fitting {0} model parameters with probability".format(modelname), ind=4)
    else:
        if estimator is not None:
            info("Fitting for {0} model parameters".format(modelname), ind=2)
            estimator = estimator.fit(X_train, y_train)
            info("Finished fitting {0} model parameters".format(modelname), ind=4)
        else:
            error("No model with name {0} was trained".format(modelname))


    return estimator



###########################################################################
#
# Tune Model
#
###########################################################################
def tuneModel(modelname, estimator, params, X_train, y_train, config):  
    info("Tuning a {0} estimator".format(modelname), ind=0)
    
    if estimator is None or params is None:
        error("There is no estimator with parameters information.", ind=2)
        return {"estimator": None, "params": None, "cv": None}

    problemType    = config['problem']
    try:
        modelData = getModelData(config, modelname)
    except:
        error("There is no model parameter data for the {0} estimator".format(modelname))

    if isClassification(problemType):
        scorers = ["accuracy", "average_precision", "f1", "f1_micro",
                   "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                   "precision", "recall", "roc_auc"]
        scorer = "roc_auc"
    
    if isClustering(problemType):
        scorers = ["adjusted_mutual_info_score", "adjusted_rand_score",
                   "completeness_score", "fowlkes_mallows_score",
                   "homogeneity_score", "mutual_info_score",
                   "normalized_mutual_info_score", "v_measure_score"]
        scorer = "adjusted_mutual_info_score"
    
    if isRegression(problemType):
        scorers = ["explained_variance", "neg_mean_absolute_error",
                   "neg_mean_squared_error", "neg_mean_squared_log_error",
                   "neg_median_absolute_error", "r2"]
        scorer = "neg_mean_absolute_error"

    if scorer not in scorers:
        raise ValueError("Scorer {0} is not allowed".format(scorer))

    searchType = "random"    
    if searchType == "grid":
        param_grid = params['grid']
        tuneEstimator = GridSearchCV(estimator, param_grid=param_grid, cv=2,
                                     scoring=scorer, verbose=1)
    elif searchType == "random":        
        n_iter_search = modelData.get('iter')
        if n_iter_search is None:
            n_iter_search = 10
        param_dist = params['dist']
        tuneEstimator = RandomizedSearchCV(estimator, param_distributions=param_dist, 
                                           cv=2, n_iter=n_iter_search, 
                                           verbose=1, n_jobs=-1,
                                           return_train_score=True)
    else:
        raise ValueError("Search type {0} is not allowed".format(searchType))


    info("Running {0} parameter search".format(searchType), ind=2)        
    tuneEstimator.fit(X_train, y_train)
    bestEstimator = tuneEstimator.best_estimator_        
    bestScore     = tuneEstimator.best_score_
    bestParams    = tuneEstimator.best_params_
    cvResults     = tuneEstimator.cv_results_
    cvScores      = cvResults['mean_test_score']
    fitTimes      = cvResults['mean_fit_time']

    info("Tested {0} Parameter Sets".format(len(fitTimes)), ind=4)
    info("CV Fit Time Info (Mean,Std): ({0} , {1})".format(round(fitTimes.mean(),1), round(fitTimes.std(),1)), ind=4)
    info("Best Score                 : {0}".format(round(bestScore, 3)), ind=4)
    info("CV Test Scores (Mean,Std)  : ({0} , {1})".format(round(cvScores.mean(),1), round(cvScores.std(),1)), ind=4)
    info("Best Parameters", ind=4)
    for paramName, paramVal in bestParams.iteritems():
        info("Param: {0} = {1}".format(paramName, paramVal), ind=6)
    

    return {"estimator": bestEstimator, "params": bestParams, "cv": cvResults}



###########################################################################
#
# Test Model
#
###########################################################################
def testModel(modelname, estimator, X_test, config):  
    info("Testing a {0} estimator".format(modelname), ind=0)
    info("X data is {0}".format(getDim(X_test)), ind=2)
    
    problemType = config['problem']
    results = {"good": True, "label": None, "prob": None, "pred": None}
    
    if isinstance(estimator, dict):
        estimator = estimator['estimator']
        
    
    if estimator is None:
        error("The {0} estimator is NULL".format(modelname))
        results['good'] = False
        return results
    
    
    if isClassification(problemType):
        info("Predicting classification labels/classes for {0}".format(modelname), ind=4)
        try:
            results['label'] = estimator.predict(X_test)
        except:
            results['good'] = False
            error("There is a problem getting labels for {0}".format(modelname), ind=4)
        
        info("Predicting classification probabilities for {0}".format(modelname), ind=4)
        try:
            proba = estimator.predict_proba(X_test)
            results['prob'] = proba[:,1]
        except:
            results['good'] = False
            error("There is a problem getting probabilities for {0}".format(modelname), ind=4)
            

    if isRegression(problemType):
        info("Predicting regression score/output for {0}".format(modelname), ind=4)
        try:
            results['pred'] = estimator.predict(X_test)
        except:
            results['good'] = False
            error("There is a problem getting prediction for {0}".format(modelname), ind=4)


    if results['good'] == True:
        info("Everything looks good for the {0} estimator".format(modelname), ind=4)
    else:        
        info("There is a problem with the {0} estimator".format(modelname), ind=4)


    return results



###########################################################################
#
# Save Model
#
###########################################################################
def getModelFileName(config, modelname):
    prefix = None
    if isRegression(config['problem']):
        prefix = "regressor"
    elif isClassification(config['problem']):
        prefix = "classifier"
    modelFileName = setFile(getModelsDir(config), "{0}-{1}.p".format(prefix,modelname))
    
    return modelFileName


def saveTrainedModel(config, modelResults):
    modelname = modelResults['name']
    modelFileName = getModelFileName(config, modelname)
    
    if modelname == "tpot":
        tpotObj = modelResults['estimator']
        tpotFileName = modelFileName.replace(".p", ".py")
        tpotObj.export(tpotFileName)
        del modelResults['estimator']
        saveJoblib(jlfile=modelFileName, jldata=modelResults, compress=True)
    else:
        saveJoblib(jlfile=modelFileName, jldata=modelResults, compress=True)


def getTrainedModel(config, modelname):
    modelFileName = getModelFileName(config, modelname)    
    modelResults  = getJoblib(jlfile=modelFileName)
    return modelResults
        