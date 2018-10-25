#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:35:19 2018

@author: tgadfort
"""


def grid_search(estimator, param_grid, verbose, scoring, cv, X, y, n_jobs=1):
    grid_search = GridSearchCV(
            estimator, 
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=2 if verbose > 0 else 0)


    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_parameters_`` and
    # ``gs.best_index_``    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.grid_scores_


def tuneParameters(estimator, config)
            estimator, best_params, grid_scores = grid_search(estimator, conf_model['params_to_search'] \
                , verbose, eval_metric, n_folds, x_train, y_train, n_jobs=n_jobs)
