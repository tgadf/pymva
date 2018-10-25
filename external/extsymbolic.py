#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:53:12 2018

@author: tgadfort
"""

from logger import info, error
from paramHelper import getParams

from gplearn.genetic import SymbolicRegressor

###############################################################################
#
# Symbolic Regression
#
#  http://gplearn.readthedocs.io/en/stable/reference.html
#
###############################################################################
def createSymbolicRegression(params = None):
    ## No Params
    srParams   = SymbolicRegressor().get_params()
    if params is None:
        params = srParams
        
        
    generations = getParams('generations', int, None, params, srParams)
    n_jobs = getParams('n_jobs', int, None, params, srParams)
    
#  = getParams('const_range': (-1.0, 1.0),
#  = getParams('function_set': ('add', 'sub', 'mul', 'div'),
#  = getParams('generations': 20,
#  = getParams('init_depth': (2, 6),
#  = getParams('init_method': 'half and half',
#  = getParams('max_samples': 1.0,
#  = getParams('metric': 'mean absolute error',
#  = getParams('n_jobs': 1,
#  = getParams('p_crossover': 0.9,
#  = getParams('p_hoist_mutation': 0.01,
#  = getParams('p_point_mutation': 0.01,
#  = getParams('p_point_replace': 0.05,
#  = getParams('p_subtree_mutation': 0.01,
#  = getParams('parsimony_coefficient': 0.001,
#  = getParams('population_size': 1000,
#  = getParams('random_state': None,
#  = getParams('stopping_criteria': 0.0,
#  = getParams('tournament_size': 20,
#  = getParams('verbose': 0,
#  = getParams('warm_start': False
    
    
    ## Estimator
    reg = SymbolicRegressor(population_size=5000,
                           generations=generations, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           n_jobs=n_jobs,
                           parsimony_coefficient=0.01, random_state=0)


    return reg

#
#est_tree = DecisionTreeRegressor()
#est_tree.fit(X_train, y_train)
#est_rf = RandomForestRegressor()
#est_rf.fit(X_train, y_train)
#
#y_gp = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
#score_gp = est_gp.score(X_test, y_test)
#y_tree = est_tree.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
#score_tree = est_tree.score(X_test, y_test)
#y_rf = est_rf.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
#score_rf = est_rf.score(X_test, y_test)
#
#fig = plt.figure(figsize=(12, 10))
#
#for i, (y, score, title) in enumerate([(y_truth, None, "Ground Truth"),
#                                       (y_gp, score_gp, "SymbolicRegressor"),
#                                       (y_tree, score_tree, "DecisionTreeRegressor"),
#                                       (y_rf, score_rf, "RandomForestRegressor")]):
#
#    ax = fig.add_subplot(2, 2, i+1, projection='3d')
#    ax.set_xlim(-1, 1)
#    ax.set_ylim(-1, 1)
#    surf = ax.plot_surface(x0, x1, y, rstride=1, cstride=1, color='green', alpha=0.5)
#    points = ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
#    if score is not None:
#        score = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score, 'x', fontsize=14)
#    plt.title(title)
#plt.show()
#
#
#graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
#Image(graph.create_png())