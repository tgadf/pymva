from __future__ import print_function, division

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split
# Statsmodels
import statsmodels.api as sm
from scipy import sparse

from .utils import import_data, generate_feature_importance_report, export_object, write_colnames
from .utils import transform_with_selected_features, collect_item_in_results, write_dataframe
from .utils import get_scorer, concat_reports_horizontally, select_initial_features, get_feature_selector
from .utils import Timer

###############################################################################
# Copied from scikit-learn package                                            #
###############################################################################
from collections import Sized

from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import check_cv, _fit_and_score, _safe_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable


def grid_search_early_stopping(estimator, param_grid, verbose, scoring, cv, X, y, early_stopping_rounds, eval_set_size, n_jobs=1, iid=True, refit=True, pre_dispatch='2*n_jobs', error_score='raise'):
    ''' This is from scikit-learn package.
    '''

    parameter_iterable = ParameterGrid(param_grid)
    scorer_ = check_scoring(estimator, scoring=scoring)

    n_samples = _num_samples(X)
    X, y = indexable(X, y)

    if y is not None:
        if len(y) != n_samples:
            raise ValueError('Target variable (y) has a different number '
                             'of samples (%i) than data (X: %i samples)'
                             % (len(y), n_samples))
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

    if verbose > 0:
        if isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(len(cv), n_candidates,
                                     n_candidates * len(cv)))

    base_estimator = clone(estimator)

    pre_dispatch = pre_dispatch

    out = Parallel(
        n_jobs=n_jobs, verbose=2 if verbose > 0 else 0,
        pre_dispatch=pre_dispatch
    )(
        delayed(_fit_and_score)(clone(base_estimator), X, y, scorer_,
                                train, test, 2 if verbose > 0 else 0, parameters,
                                {
                                    "early_stopping_rounds": early_stopping_rounds,
                                    "eval_metric": get_xgboost_eval_metric(scoring),
                                    "eval_set": [_safe_split(estimator, X, y, test, train)],
                                    "verbose": True if verbose > 1 else False
                                },                          
                                return_parameters=True,
                                error_score=error_score)
            for parameters in parameter_iterable
            for train, test in cv)

    # Out is a list of triplet: score, estimator, n_test_samples
    n_fits = len(out)
    n_folds = len(cv)

    scores = list()
    grid_scores = list()
    for grid_start in range(0, n_fits, n_folds):
        n_test_samples = 0
        score = 0
        all_scores = []
        for this_score, this_n_test_samples, _, parameters in \
                out[grid_start:grid_start + n_folds]:
            all_scores.append(this_score)
            if iid:
                this_score *= this_n_test_samples
                n_test_samples += this_n_test_samples
            score += this_score
        if iid:
            score /= float(n_test_samples)
        else:
            score /= float(n_folds)
        scores.append((score, parameters))
        # TODO: shall we also store the test_fold_sizes?
        grid_scores.append(_CVScoreTuple(
            parameters,
            score,
            np.array(all_scores)))

    # Find the best parameters by comparing on the mean validation score:
    # note that `sorted` is deterministic in the way it breaks ties
    best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
    best_score_ = best.mean_validation_score

    if refit:
        # fit the best estimator using the entire dataset
        # clone first to work around broken estimators
        best_estimator = clone(base_estimator).set_params(
            **best.parameters)

        if y is not None:
            best_estimator, _, _ = fit_estimator_early_stopping(best_estimator, X, y, scoring, early_stopping_rounds, eval_set_size, verbose)
        else:
            raise ValueError('y is required.')

    return best_estimator, best.parameters, grid_scores

###############################################################################

def select_final_features(colnames, feature_importance, params):
    df = generate_feature_importance_report(colnames, feature_importance)

    feature_selector = get_feature_selector(params['which'])

    important_features = feature_selector(df, params['threshold'])

    return important_features.loc[:, 'colname'].values.tolist()


def check_conf_model(conf_model):
    if 'params' not in conf_model:
        conf_model['params'] = {}

    return conf_model


def average_precision_score_xgboost(y_predicted, dtrain):
    return 'average_precision', -average_precision_score(dtrain.get_label(), y_predicted)


def get_xgboost_eval_metric(eval_metric):
    xgboost_eval_metrics = {
        'roc_auc': 'auc',
        'average_precision': average_precision_score_xgboost,
        'log_loss': 'logloss'
    }

    return xgboost_eval_metrics[eval_metric]


def get_xgboost_feature_importance(estimator, features_count):
    ''' This is from xgboost package.
    '''

    fs = estimator.booster().get_fscore()
    keys = [int(k.replace('f', '')) for k in fs.keys()]
    fs_dict = dict(zip(keys, fs.values()))
    all_features_dict = dict.fromkeys(list(range(features_count)), 0)
    all_features_dict.update(fs_dict)

    feature_importance = np.array(all_features_dict.values())

    return feature_importance / feature_importance.sum()


def grid_search(estimator, param_grid, verbose, scoring, cv, X, y, n_jobs=1):
    grid_search = GridSearchCV(
        estimator, 
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=2 if verbose > 0 else 0
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.grid_scores_


def fit_estimator(estimator, X, y):
    estimator.fit(X, y)
    return estimator, None, None


def fit_estimator_early_stopping(estimator, X, y, eval_metric, early_stopping_rounds, eval_set_size, verbose):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=eval_set_size, random_state=estimator.get_params()['seed'], stratify=y)
    estimator.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=get_xgboost_eval_metric(eval_metric), eval_set=[(X_valid, y_valid)], verbose=True if verbose > 1 else False)

    return estimator, None, None


def get_final_score(estimator, x_train, y_train, model_type, n_jobs, n_folds, eval_metric):
    y_predicted = estimator.predict_proba(x_train)[:, 1]


    scorer = get_scorer(eval_metric)
    score = scorer(y_train, y_predicted)
    
    cv_score = cross_val_score(estimator, x_train, y_train, scoring=eval_metric, cv=n_folds, n_jobs=n_jobs, verbose=0).mean()

    return pd.DataFrame([{'score': score, 'cv_score': cv_score}], columns=['score', 'cv_score']), y_predicted


def grid_scores_to_dataframe(grid_scores):
    rows = []
    
    if grid_scores == None:
        columns = []
    else:
        
        for grid_score in grid_scores:
            row = grid_score.parameters.copy()
            row['cv_score'] = grid_score.mean_validation_score
            row['cv_score_std'] = grid_score.cv_validation_scores.std()
            rows.append(row.copy())

        columns = grid_scores[0].parameters.keys()
        columns.extend(['cv_score', 'cv_score_std'])
   
    return pd.DataFrame(rows, columns=columns)


def return_build_output(scores, best_params, grid_scores, colnames, feature_importance, estimator, y_predicted):
    return {
        'scores':scores, 
        'best_params': pd.DataFrame([best_params]),
        'grid_scores': grid_scores_to_dataframe(grid_scores),
        'feature_importance': generate_feature_importance_report(colnames, feature_importance),
        'estimator': estimator,
        'colnames': colnames,
        'y_predicted': y_predicted
    }
    

def build_logistic(x_train, y_train, colnames, verbose, n_jobs, random_state, n_folds, eval_metric, conf_model, **kwargs):
    # TODO
    # Cap outliers
    # Scale to unit variance
    # Score in parallel
    
    # If using sk.SGD then params_to_search labeled as alpha
    # If using sk.LogisticRegression then params_to_search labeled as C

    conf_model = check_conf_model(conf_model)

    if conf_model['method'] == 'logit_sgd':
        estimator = SGDClassifier(
            loss='log',
            random_state=random_state,
            verbose=0,
            n_jobs=1,
            warm_start=False,
            **conf_model['params']
        )
    elif conf_model['method'] == 'logit':
        estimator = LogisticRegression(
            random_state = random_state,
            verbose = 0,
            n_jobs = 1,
            warm_start = False,
            **conf_model['params']
        )
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)

    if 'params_to_search' in conf_model:
        with Timer('fitting with grid search', **kwargs):
            estimator, best_params, grid_scores = grid_search(estimator, conf_model['params_to_search'] \
                , verbose, eval_metric, n_folds, x_train, y_train, n_jobs=n_jobs)
    else:
        with Timer('fitting with fixed parameters', **kwargs):
            estimator, best_params, grid_scores = fit_estimator(estimator, x_train, y_train)

    if 'feature_selection' in conf_model:
        raise ValueError('Final feature selection for logistic regression models is not supported. Remove "feature_selection" option from logisitc regression model.')
        # if conf_model['feature_selection']['which'] not in ['at_least', 'top']:
        #     raise ValueError('Only "at_least" or "top" is supported as feature selection method for logistic regression.')
        #
        # with Timer('selecting final important features', **kwargs):
        #     selected_colnames = select_final_features(colnames, estimator.coef_[0,:], conf_model['feature_selection'])
        #     x_train, colnames = transform_with_selected_features(x_train, colnames, selected_colnames)
        # with Timer('fitting final model', **kwargs):
        #     # TODO should we force no regulirization here?
        #     estimator.set_params(penalty='none')
        #     estimator, _, _ = fit_estimator(estimator, x_train, y_train)

    with Timer('scoring train and cross validation data', **kwargs):
        scores, y_predicted = get_final_score(estimator, x_train, y_train, conf_model['method'], n_jobs, n_folds, eval_metric)

    return return_build_output(scores, best_params, grid_scores, colnames, estimator.coef_[0,:], estimator, y_predicted)


def build_random_forest(x_train, y_train, colnames, verbose, n_jobs, random_state, n_folds, eval_metric, conf_model, **kwargs):
    conf_model = check_conf_model(conf_model)

    estimator = RandomForestClassifier(
        oob_score=False, 
        warm_start=False, 
        class_weight=None, 
        verbose=0, 
        n_jobs=n_jobs, 
        random_state=random_state,
        **conf_model['params']
    )

    if 'params_to_search' in conf_model:
        with Timer('fitting with grid search', **kwargs):
            estimator, best_params, grid_scores = grid_search(estimator, conf_model['params_to_search'], verbose, eval_metric, n_folds, x_train, y_train)
    else:
        with Timer('fitting with fixed parameters', **kwargs):
            estimator, best_params, grid_scores = fit_estimator(estimator, x_train, y_train)

    if 'feature_selection' in conf_model:
        with Timer('selecting final important features', **kwargs):
            selected_colnames = select_final_features(colnames, estimator.feature_importances_, conf_model['feature_selection'])
            x_train, colnames = transform_with_selected_features(x_train, colnames, selected_colnames)

        with Timer('fitting final model', **kwargs):
            estimator, _, _ = fit_estimator(estimator, x_train, y_train)
    
    with Timer('scoring train and cross validation data', **kwargs):
        scores, y_predicted = get_final_score(estimator, x_train, y_train, conf_model['method'], 1, n_folds, eval_metric)

    return return_build_output(scores, best_params, grid_scores, colnames, estimator.feature_importances_, estimator, y_predicted)


def build_xgboost(x_train, y_train, colnames, verbose, n_jobs, random_state, n_folds, eval_metric, conf_model, **kwargs):
    conf_model = check_conf_model(conf_model)

    estimator = XGBClassifier(
        silent=True,
        objective='binary:logistic',
        nthread=n_jobs,
        seed=0,
        **conf_model['params']
    )

    if 'params_to_search' in conf_model:
        with Timer('fitting with grid search', **kwargs):
            if 'early_stopping' in conf_model and 'during_grid_search' in conf_model['early_stopping'] and conf_model['early_stopping']['during_grid_search']:
                estimator, best_params, grid_scores = grid_search_early_stopping(
                    estimator, conf_model['params_to_search'], verbose, eval_metric, n_folds, x_train, y_train, 
                    conf_model['early_stopping']['early_stopping_rounds'], conf_model['early_stopping']['eval_set_size']
                )
            else:
                estimator, best_params, grid_scores = grid_search(estimator, conf_model['params_to_search'], verbose, eval_metric, n_folds, x_train, y_train)
    else:
        with Timer('fitting with fixed parameters', **kwargs):
            if 'early_stopping' in conf_model:
                estimator, best_params, grid_scores = fit_estimator_early_stopping(
                    estimator, x_train, y_train,
                    eval_metric, conf_model['early_stopping']['early_stopping_rounds'], conf_model['early_stopping']['eval_set_size'], verbose
                )
            else:        
                estimator, best_params, grid_scores = fit_estimator(estimator, x_train, y_train)

    if 'feature_selection' in conf_model:
        with Timer('selecting final important features', **kwargs):
            selected_colnames = select_final_features(colnames, get_xgboost_feature_importance(estimator, len(colnames)), conf_model['feature_selection'])
            x_train, colnames = transform_with_selected_features(x_train, colnames, selected_colnames)

        with Timer('fitting final model', **kwargs):
            if 'early_stopping' in conf_model:
                estimator, _, _ = fit_estimator_early_stopping(
                    estimator, x_train, y_train, 
                    eval_metric, conf_model['early_stopping']['early_stopping_rounds'], conf_model['early_stopping']['eval_set_size'], verbose
                )
            else:        
                estimator, _, _ = fit_estimator(estimator, x_train, y_train)

    with Timer('scoring train and cross validation data', **kwargs):
        scores, y_predicted = get_final_score(estimator, x_train, y_train, conf_model['method'], 1, n_folds, eval_metric)

    return return_build_output(scores, best_params, grid_scores, colnames, get_xgboost_feature_importance(estimator, len(colnames)), estimator, y_predicted)


def get_builder(method):
    builders = {
        'logit': build_logistic,
        'logit_sgd': build_logistic,
        'random_forest': build_random_forest,
        'xgboost': build_xgboost
    }

    return builders[method]

def compare_sm_logistic(x_data, y_data, features, final_estimator, penalty, params, **kwargs):
    if sparse.issparse(x_data) == True:
        numpy_dense = x_data.todense()
        x_data = pd.DataFrame(numpy_dense)
    else:
        x_data = pd.DataFrame(x_data)
    x_data.index = y_data.index
    
    # Align columns with Col_names
    x_data.columns = features
    sm_logit_fit = None
    if final_estimator.fit_intercept == True:
        x_data = sm.add_constant(x_data, prepend = False)
    sm_logit = sm.Logit(y_data, x_data)
    try:
    # Fit the statsmodels logit for parameter evaluation
        if penalty == 'l1':
            is_penalty = '\n l1 penalty C: ' + str(params) + '\n'
            sm_logit_fit = sm_logit.fit_regularized(disp = False, alpha = 1.0/params, method = 'l1')
        else:
            is_penalty = '\n No Penalty  C: ' + str(params) + '\n'
            sm_logit_fit = sm_logit.fit()
    except:
        pass
    # Return model summary
    if sm_logit_fit == None:
        model_comparison = 'Model Error'
    else:
        model_comparison = is_penalty + '\n' + str(sm_logit_fit.summary()) + '\n'
    return model_comparison 

def pretty_print_logistic_coeff(coeffs, intercept, features):
    # takes the model features and the coefficients from scikit lear LM 
    # and prints them in pretty_print format
    mod_coef = coeffs[0].tolist() 
    max_len = max([len(x) for x in features])
    coef_dict = OrderedDict(zip(features, mod_coef))
    if intercept != 0 : coef_dict['Intercept'] = intercept[0]
    else: coef_dict['Intercept'] = intercept
    #coef_dict['Intercept'] = intercept[0]
    clen = 10 # Defines the coeffient precision to print
    coef_values = ''
    for k in (coef_dict):
        if float(coef_dict[k]) < 0:
            coef= str(coef_dict[k])[0:clen]
            coef_values += str(k) + ':'+ ' '*(max_len + 4 - len(k))  + coef + '\n'
        else:
            coef= str(coef_dict[k])[0:clen-1]
            coef_values += str(k) + ':'+ ' '*(max_len + 5 - len(k))  + coef + '\n'
    return coef_values


def export_model_and_prediction(path_to_output, model_name, target, method, estimator, colnames, y_predicted, config, x_data, y_data, **kwargs):
    model = {
        'target': target,
        'method': method,
        'estimator': estimator,
        'colnames': colnames
    }

    # output optional statsmodels logit parameters
    if str(method) in ['logit', 'logit_sgd']:
        if str(method) == 'logit_sgd':
            mod_param = estimator.get_params()['alpha']
        else:
            mod_param = estimator.get_params()['C']
        mod_penalty = estimator.get_params()['penalty']

        # Export current model coefficients and statsmodels summary
        model_coef = '\n Scikit Learn ModelCoefficients & Parameters: \n' \
                + 'Regularization: ' + str(mod_penalty) + ', penalty: ' + str(mod_param) +  '\n\n' \
                + pretty_print_logistic_coeff(estimator.coef_, estimator.intercept_, colnames) + '\n'
                
        export_object(model_coef, os.path.join(path_to_output, 'models', model_name, target, 'coefficients'), 'coefficients.txt', **kwargs)
        ###
        #if 'comparison' == True:
        model_comparison = compare_sm_logistic(x_data, y_data, colnames, estimator, penalty = mod_penalty, params = mod_param)

        model_comparison_output = model_coef + '\n Stats_Models Comparisons\n' + str(target) + '\n' + model_comparison
        export_object(model_comparison_output, os.path.join(path_to_output, 'models', model_name, target, 'model_comparisons') \
            , 'model_parameters.txt', **kwargs)


    export_object(model, os.path.join(path_to_output, 'models', model_name, target), 'model.p', **kwargs)
    write_colnames(colnames, os.path.join(path_to_output, 'models', model_name, target), 'colnames.txt', **kwargs)
    
    export_object(y_predicted, os.path.join(path_to_output, 'predicted', model_name, target, 'train'), 'y_train_predicted.p', **kwargs)


def get_colnames_for_scoring(results, best_models):
    best_model_names = best_models.set_index('target')['best_model']

    colnames_for_scoring = []
    for target, result in results.iteritems():
        colnames_for_scoring.extend(result[best_model_names[target]]['colnames'])

    colnames_for_scoring = [colname.split(':')[0] if colname.startswith('CAT_') else colname for colname in colnames_for_scoring]

    return list(set(colnames_for_scoring))


def export_model_reports(results, path_to_output, model_names, **kwargs):
    df_scores_all = []
    
    for model_name in model_names:
        df_scores = collect_item_in_results(results, model_name, 'scores')
        write_dataframe(df_scores, os.path.join(path_to_output, 'reports', model_name), 'scores_{0}.csv'.format(model_name), **kwargs)
        
        df_scores_all.append(df_scores)

        df_best_params = collect_item_in_results(results, model_name, 'best_params')
        write_dataframe(df_best_params, os.path.join(path_to_output, 'reports', model_name), 'best_params_{0}.csv'.format(model_name), **kwargs)  

        df_grid_scores = collect_item_in_results(results, model_name, 'grid_scores')
        write_dataframe(df_grid_scores, os.path.join(path_to_output, 'reports', model_name), 'grid_scores_{0}.csv'.format(model_name), **kwargs)  

        df_feature_importance = collect_item_in_results(results, model_name, 'feature_importance')
        write_dataframe(df_feature_importance, os.path.join(path_to_output, 'reports', model_name), 'feature_importance_{0}.csv'.format(model_name), **kwargs) 

    df_scores_all = concat_reports_horizontally(df_scores_all)
    
    cv_score_columns = [colname for colname in df_scores_all.columns if colname.endswith('cv_score')]
    
    write_dataframe(df_scores_all, os.path.join(path_to_output, 'reports'), 'scores_all.csv', **kwargs)
     
    best_models = df_scores_all[['target']].copy()
    best_models['best_model'] = df_scores_all[cv_score_columns].idxmax(axis=1).apply(lambda x: x.replace('.cv_score', ''))
    write_dataframe(best_models, os.path.join(path_to_output, 'scoring'), 'best_models.csv', **kwargs)         

    colnames_for_scoring = get_colnames_for_scoring(results, best_models)
    write_colnames(colnames_for_scoring, os.path.join(path_to_output, 'scoring'), 'colnames_for_scoring.txt', **kwargs)

def build(conf):
    with Timer('[build] command'):
        if 'feature_selection' in conf:
            selected_colnames_all = select_initial_features(conf['path_to_output'], conf['feature_selection'], conf['targets'], log_start_time=False, indent=2, level='debug')
        else:
            selected_colnames_all = {target: None for target in conf['targets']}
        def build_models(target, selected_colnames):
            with Timer('building models for [{0}]'.format(target), indent=4):
                with Timer('importing train data', indent=6):
                    x_train, y_train, colnames = import_data(conf['path_to_output'], target, 'train', log_start_time=False, indent=8, level='debug')
                
                if selected_colnames is not None:
                    with Timer('selecting important features', indent=6):
                        x_train, colnames = transform_with_selected_features(x_train, colnames, selected_colnames)
                result = {}

                for model_name in conf['models']:
                    if conf['models'][model_name]['method'] != 'ensemble':
                        with Timer('building [{0}] model'.format(model_name), indent=6):
                            builder = get_builder(conf['models'][model_name]['method'])
                            
                            output = builder(x_train, y_train, colnames, conf['verbose'], conf['n_jobs'] \
                                    , conf['random_state'], conf['n_folds'], conf['eval_metric'], conf['models'][model_name], indent=8)

                        with Timer('exporting [{0}] model and predictions'.format(model_name), indent=6):
                            export_model_and_prediction(conf['path_to_output'], model_name, target, conf['models'][model_name]['method'] \
                                    , output['estimator'], output['colnames'], output['y_predicted'], conf \
                                    # Optional logit data
                                    , x_data = x_train, y_data = y_train \
                                    , log_start_time=False, indent=8, level='debug')
                        result[model_name] = output
                        
                for model_name in conf['models']:
                    if conf['models'][model_name]['method'] == 'ensemble':
                        # TODO
                        pass
            return result
            
        with Timer('building models for all targets', indent=2):
            results = {target: build_models(target, selected_colnames_all[target]) for target in conf['targets']}

        with Timer('exporting model reports', indent=2):
            export_model_reports(results, conf['path_to_output'], conf['models'].keys(), log_start_time=False, indent=4, level='debug')






