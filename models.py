#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:06:46 2018

@author: tgadfort
"""

from logger import info
from targetInfo import isClassification,isRegression

from classifierBase import classifier

##### Regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge, ARDRegression, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import HuberRegressor,TheilSenRegressor,RANSACRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR, NuSVR, LinearSVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.isotonic import IsotonicRegression

from pyearth import Earth

from gplearn.genetic import SymbolicRegressor

from tpot import TPOTRegressor

##### Classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier

from tpot import TPOTClassifier

import external.extearth, external.extsymbolic, external.extlightning
import external.exttpot


###########################################################################
#
# Model Types
#
###########################################################################
def getModelType(modelName):
    if modelName in ['linear', 'logistic', 'linear', 'ridge', 'lasso',
                     'elasticnet', 'omp', 'bayesridge', 'ard', 'sgd',
                     'passagg', 'perceptron', 'huber', 'theilsen', 'ransac']:
        return "linear"
    elif modelName in ["rf", "extratrees", "gbm", "adaboost", "xgboost"]:
        return "ensemble"
    elif modelName in ["mlp"]:
        return "nn"
    elif modelName in ["nb", "nbbern", "nbmulti"]:
        return "nb"
    elif modelName in ["svmlin", "svmnupoly", "svmnulinear", "svmnusigmoid",
                       "svmnurbf", "svmepspoly", "svmepslinear",
                       "svmepssigmoid", "svmepsrbf"]:
        return "svm"
    elif modelName in ["lda", "qda"]:
        return "discrim"
    elif modelName in ["dtree"]:
        return "tree"
    elif modelName in ["earth", "tpot"]:
        return "external"
    else:
        raise ValueError("Model name {0} not recognized".format(modelName))
        
    return None


###########################################################################
#
# Models
#
###########################################################################
def getModels(config, level):
    info("Getting Models For Level {0}".format(level),ind=0)
    problemType = config['problem']
    
    if isClassification(problemType):
        models0 = ["xgboost", "logistic"]
        models1 = ["rf", "nn", "svmnulinear", "gbm"]
        models2 = ["extratrees", "sgd", "nb", "lda", "kneighbors", "svmepslinear"]
        models3 = ["passagg", "gaussproc", "qda", "nbbern", "nbmulti",
                   "dtree", "rneighbors", "svmlin", "svmnu", "adaboost",
                   "svmnupoly", "svmepspoly", "svmnusigmoid", "svmepssigmoid",
                   "svmnurbf", "svmepsrbf"]
        if level == 0:
            models = models0
        elif level == 1:
            models = models0 + models1
        elif level == 2:
            models = models0 + models1 + models2
        else:
            models = models0 + models1 + models2 + models3
            
    
    if isRegression(problemType):
        models0 = ["xgboost", "linear"]
        models1 = ["xgboost", "rf", "linear", "nn", "svm", "gbm"]
        models2 = ["extratrees", "sgd", "earth", "kernelridge", "kneighbors"]
        models3 = ["passagg", "gaussproc", "lasso", "ridge", "elasticnet",
                   "bayesridge", "huber", "theilsen", "ransac",
                   "dtree", "rneighbors", "svmlin", "svmnu", "adaboost"]
        if level == 0:
            models = models0
        elif level == 1:
            models = models0 + models1
        elif level == 2:
            models = models0 + models1 + models2
        else:
            models = models0 + models1 + models2 + models3
            
    
    info("Using the following models: {0}".format(models), 2)
    return models



def getModelData(config, modelname):
    problemType    = config['problem']
        
    try:
        modelType   = getModelType(modelname)
        modelData   = config['models'][problemType][modelType][modelname]
    except:
        info("No model parameters for {0}".format(modelname))
        modelData = {}

    return modelData
    


def getModel(config, modelname):
    info("Getting {0} Model".format(modelname),ind=0)
    
    problemType = config['problem']
    modelData   = getModelData(config, modelname)
    modelParams = modelData.get('params')
    retval      = None


    ###########################################################################
    # Classification
    ###########################################################################
    if isClassification(problemType):        
        if modelname == "logistic":
            retval = classifier(modelname, LogisticRegression(), modelParams)
        if modelname == "sgd":
            retval = classifier(modelname, SGDClassifier(), modelParams)
        if modelname == "passagg":
            retval = classifier(modelname, PassiveAggressiveClassifier(), modelParams)
    
        if modelname == "mlp":
            retval = classifier(modelname, MLPClassifier(), modelParams)
    
        if modelname == "xgboost":
            retval = classifier(modelname, XGBClassifier(), modelParams)
    
        if modelname == "gaussproc":
            retval = classifier(modelname, GaussianProcessClassifier(), modelParams)
    
        if modelname == "lda":
            retval = classifier(modelname, LinearDiscriminantAnalysis(), modelParams)
        if modelname == "qda":
            retval = classifier(modelname, QuadraticDiscriminantAnalysis(), modelParams)
            
        if modelname == "nb":
            retval = classifier(modelname, GaussianNB(), modelParams)
        if modelname == "nbbern":
            retval = classifier(modelname, BernoulliNB(), modelParams)
        if modelname == "nbmulti":
            retval = classifier(modelname, MultinomialNB(), modelParams)
    
        if modelname == "dtree":
            retval = classifier(modelname, DecisionTreeClassifier(), modelParams)
            
        if modelname == "kneighbors":
            retval = classifier(modelname, KNeighborsClassifier(), modelParams)
        if modelname == "rneighbors":
            retval = classifier(modelname, RadiusNeighborsClassifier(), modelParams)
                
        if modelname == "svmlin":
            retval = classifier(modelname, LinearSVC(), modelParams)
        if modelname == "svmnupoly":
            retval = classifier(modelname, NuSVC(), modelParams)
        if modelname == "svmnulinear":
            retval = classifier(modelname, NuSVC(), modelParams)
        if modelname == "svmnusigmoid":
            retval = classifier(modelname, NuSVC(), modelParams)
        if modelname == "svmnurbf":
            retval = classifier(modelname, NuSVC(), modelParams)
        if modelname == "svmepspoly":
            retval = classifier(modelname, SVC(), modelParams)
        if modelname == "svmepslinear":
            retval = classifier(modelname, SVC(), modelParams)
        if modelname == "svmepssigmoid":
            retval = classifier(modelname, SVC(), modelParams)
        if modelname == "svmepsrbf":
            retval = classifier(modelname, SVC(), modelParams)

        if modelname == "rf":
            retval = classifier(modelname, RandomForestClassifier(), modelParams)
        if modelname == "extratrees":
            retval = classifier(modelname, ExtraTreesClassifier(), modelParams)
        if modelname == "adaboost":
            retval = classifier(modelname, AdaBoostClassifier(), modelParams)
        if modelname == "gbm":
            retval = classifier(modelname, GradientBoostingClassifier(), modelParams)
            
        if modelname == "tpot":
            retval = classifier(modelname, TPOTClassifier(), modelParams)


        #######################################################################
        # Regression
        #######################################################################
        if modelname == "lightning":
            retval = external.extlightning.createLightningClassifier(modelParams)



    ###########################################################################
    # Regression
    ###########################################################################
    if isRegression(problemType):
        if modelname == "linear":
            retval = classifier(modelname, LinearRegression(), modelParams)
        if modelname == "ridge":
            retval = classifier(modelname, Ridge(), modelParams)
        if modelname == "lasso":
            retval = classifier(modelname, Lasso(), modelParams)
        if modelname == "elasticnet":
            retval = classifier(modelname, ElasticNet(), modelParams)
        if modelname == "omp":
            retval = classifier(modelname,OrthogonalMatchingPursuit(), modelParams)
        if modelname == "bayesridge":
            retval = classifier(modelname,BayesianRidge(), modelParams)
        if modelname == "ard":
            retval = classifier(modelname,ARDRegression(), modelParams)
        if modelname == "sgd":
            retval = classifier(modelname,SGDRegressor(), modelParams)
        if modelname == "passagg":
            retval = classifier(modelname, PassiveAggressiveRegressor(), modelParams)
        if modelname == "perceptron":
            retval = None
        if modelname == "huber":
            retval = classifier(modelname, HuberRegressor(), modelParams)
        if modelname == "theilsen":
            retval = classifier(modelname, TheilSenRegressor(), modelParams)
        if modelname == "ransac":
            retval = classifier(modelname, RANSACRegressor(), modelParams)

        if modelname == "mlp":
            retval = classifier(modelname, MLPRegressor(), modelParams)
    
        if modelname == "xgboost":
            retval = classifier(modelname, XGBRegressor(), modelParams)

        if modelname == "gaussproc":
            retval = classifier(modelname, GaussianProcessRegressor(), modelParams)
    
        if modelname == "dtree":
            retval = classifier(modelname, DecisionTreeRegressor(), modelParams)
            
        if modelname == "kneighbors":
            retval = classifier(modelname, KNeighborsRegressor(), modelParams)
        if modelname == "rneighbors":
            retval = classifier(modelname, RadiusNeighborsRegressor(), modelParams)
            
        if modelname == "svmlin":
            retval = classifier(modelname, LinearSVR(), modelParams)
        if modelname == "svmnupoly":
            retval = classifier(modelname, NuSVR(), modelParams)
        if modelname == "svmnulinear":
            retval = classifier(modelname, NuSVR(), modelParams)
        if modelname == "svmnusigmoid":
            retval = classifier(modelname, NuSVR(), modelParams)
        if modelname == "svmnurbf":
            retval = classifier(modelname, NuSVR(), modelParams)
        if modelname == "svmepspoly":
            retval = classifier(modelname, SVR(), modelParams)
        if modelname == "svmepslinear":
            retval = classifier(modelname, SVR(), modelParams)
        if modelname == "svmepssigmoid":
            retval = classifier(modelname, SVR(), modelParams)
        if modelname == "svmepsrbf":
            retval = classifier(modelname, SVR(), modelParams)

        if modelname == "rf":
            retval = classifier(modelname, RandomForestRegressor(), modelParams)
        if modelname == "extratrees":
            retval = classifier(modelname, ExtraTreesRegressor(), modelParams)
        if modelname == "adaboost":
            retval = classifier(modelname, AdaBoostRegressor(), modelParams)
        if modelname == "gbm":
            retval = classifier(modelname, GradientBoostingRegressor(), modelParams)

        if modelname == "isotonic":
            retval = classifier(modelname, IsotonicRegression(), modelParams)
            
        if modelname == "earth":
            retval = classifier(modelname, Earth(), modelParams)
            
        if modelname == "symbolic":
            retval = classifier(modelname, SymbolicRegressor(), modelParams)
            
        if modelname == "tpot":
            retval = classifier(modelname, TPOTRegressor(), modelParams)


    if retval is None:
        raise ValueError("No model with name {0} was created".format(modelname))

    model = retval.get()

    return model