#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:43:53 2018

@author: tgadfort
"""

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

###############################################################################
#
# AutoML Classifier
#
#   pip install auto_ml
#
###############################################################################
def createAutoMLClassifier(params = None):

    df_train, df_test = get_boston_dataset()
    
    column_descriptions = {
        'MEDV': 'output', 'CHAS': 'categorical'
    }
    
    ml_predictor = Predictor(type_of_estimator='classifer', column_descriptions=column_descriptions)
    
    ml_predictor.train(df_train)
    
    #ml_predictor.train(data, model_names=['DeepLearningClassifier'])
    # Available options are
    # DeepLearningClassifier and DeepLearningRegressor
    # XGBClassifier and XGBRegressor
    # LGBMClassifer and LGBMRegressor
    # CatBoostClassifier and CatBoostRegressor
    
    ml_predictor.score(df_test, df_test.MEDV)