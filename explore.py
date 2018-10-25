import os

from sklearn.ensemble import RandomForestClassifier

from .utils import import_data, generate_feature_importance_report, collect_item_in_results, write_dataframe, Timer


def explore_random_forest(x_train, y_train, colnames, verbose, n_jobs, random_state, params):
    if verbose > 0:
        verbose = 1
        
    estimator = RandomForestClassifier(verbose=verbose, n_jobs=n_jobs, random_state=random_state, **params)

    estimator.fit(x_train, y_train)

    return {'feature_importance': generate_feature_importance_report(colnames, estimator.feature_importances_)}


def get_feature_explorer(method):
    feature_explorers = {'random_forest': explore_random_forest}

    return feature_explorers[method]


def explore(conf):
    if 'feature_importance' in conf:
        with Timer('[explore] command'):
            def explore_features(target):
                with Timer('exploring features for [{0}]'.format(target), indent=4):
                    with Timer('importing train data', indent=6):
                        x_train, y_train, colnames = import_data(conf['path_to_output'], target, 'train', log_start_time=False, indent=8, level='debug')

                    result = {}

                    for model_name in conf['feature_importance']:
                        with Timer('exploring with {0}'.format(model_name), indent=6):
                            feature_explorer = get_feature_explorer(conf['feature_importance'][model_name]['method'])
                            result[model_name] = feature_explorer(x_train, y_train, colnames, conf['verbose'], conf['n_jobs'], conf['random_state'], conf['feature_importance'][model_name]['params'])

                return result

            with Timer('exploring features for all targets', indent=2):
                results = {target: explore_features(target) for target in conf['targets']}

            with Timer('exporting feature importance reports', indent=2):
                for model_name in conf['feature_importance']:
                    feature_importance = collect_item_in_results(results, model_name, 'feature_importance', should_append_model_name_to_colnames=False)
                    write_dataframe(feature_importance, os.path.join(conf['path_to_output'], 'feature_importance'), 'feature_importance_{0}.csv'.format(model_name), log_start_time=False, indent=4, level='debug')
    else:
        with Timer('[explore] command. No method to explore features is specified', log_start_time=0):
            pass