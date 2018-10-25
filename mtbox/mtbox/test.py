import os

import pandas as pd
import numpy as np

from .utils import import_data, import_object, transform_with_selected_features, get_scorer, concat_reports_horizontally, select_initial_features, export_object, write_dataframe, collect_item_in_results, Timer, get_train_event_rate


def generate_lift_chart(y_actual, y_predicted, n_bins):
    noise = np.random.normal(scale=1.0E-8, size=y_predicted.shape)
    df = pd.DataFrame({'y_actual': y_actual, 'y_predicted': y_predicted + noise})
    
    n_events = df['y_actual'].sum()
    actual_event_rate = df['y_actual'].mean()
    
    df['bin'] = pd.qcut(-df['y_predicted'], n_bins, labels=list(range(1, n_bins + 1)))
    grouped = df.groupby('bin')
    
    df_summary = grouped.agg({'y_actual': ['count', 'sum'], 'y_predicted':['mean']})
    df_summary.columns = ['.'.join(col) for col in df_summary.columns.values]
    df_summary.rename(columns={'y_actual.count': 'n_obs', 'y_actual.sum': 'n_events', 'y_predicted.mean': 'avg_score'}, inplace=True)
    
    df_summary['captured_rate'] = df_summary['n_events'] / n_events
    df_summary['cumulative_captured_rate'] = df_summary['captured_rate'].cumsum()
    df_summary['actual_event_rate'] = df_summary['n_events'] / df_summary['n_obs']
    df_summary['cumulative_actual_event_rate'] = df_summary['n_events'].cumsum() / df_summary['n_obs'].cumsum()
    df_summary['cumulative_avg_score'] = (df_summary['avg_score'] * df_summary['n_obs']).cumsum() / df_summary['n_obs'].cumsum()
    df_summary['lift'] = df_summary['actual_event_rate'] / actual_event_rate
    df_summary['cumulative_lift'] = df_summary['cumulative_actual_event_rate'] / actual_event_rate

    df_summary = df_summary[['n_obs', 'n_events', 'captured_rate', 'cumulative_captured_rate', 'avg_score', 'actual_event_rate', 'cumulative_avg_score', 'cumulative_actual_event_rate', 'lift', 'cumulative_lift']]

    df_summary.reset_index(inplace=True)
    
    return df_summary


def test_model(model, x_test, y_test, colnames, verbose, n_jobs, eval_metric, p, r, **kwargs):
    with Timer('selecting model-specific important features', **kwargs):
        x_test, colnames = transform_with_selected_features(x_test, colnames, model['colnames'])

    with Timer('scoring test data', **kwargs):
        y_predicted = model['estimator'].predict_proba(x_test)[:, 1]
        
        # p: event rate, r: train event rate
        # if r is not None, then adjust the score
        if r:
            y_predicted = p * (1 - r) * y_predicted / (r * (1 - p) - y_predicted * (r - p))

        scorer = get_scorer(eval_metric)
        test_score = pd.DataFrame([{'test_score': scorer(y_test, y_predicted)}])

        decile_report = generate_lift_chart(y_test.values, y_predicted, 10)
        percentile_report = generate_lift_chart(y_test.values, y_predicted, 100)

    return {'test_score': test_score, 'y_predicted': y_predicted, 'decile_report': decile_report, 'percentile_report': percentile_report}


def export_test_results(results, path_to_output, model_names, **kwargs):
    df_test_score_all = []

    for model_name in model_names:
        df_test_score = collect_item_in_results(results, model_name, 'test_score')
        write_dataframe(df_test_score, os.path.join(path_to_output, 'reports', model_name), 'test_scores_{0}.csv'.format(model_name), **kwargs)

        df_decile_report = collect_item_in_results(results, model_name, 'decile_report')
        write_dataframe(df_decile_report, os.path.join(path_to_output, 'reports', model_name), 'decile_reports_{0}.csv'.format(model_name), **kwargs)  

        df_percentile_report = collect_item_in_results(results, model_name, 'percentile_report')
        write_dataframe(df_percentile_report, os.path.join(path_to_output, 'reports', model_name), 'percentile_reports_{0}.csv'.format(model_name), **kwargs)  

        df_test_score_all.append(df_test_score)

    df_test_score_all = concat_reports_horizontally(df_test_score_all)
    write_dataframe(df_test_score_all, os.path.join(path_to_output, 'reports'), 'test_scores_all.csv', **kwargs)


def test(conf):
    with Timer('[test] command'):
        if 'feature_selection' in conf:
            selected_colnames_all = select_initial_features(conf['path_to_output'], conf['feature_selection'], conf['targets'], log_start_time=False, indent=2, level='debug')
        else:
            selected_colnames_all = {target: None for target in conf['targets']}

        def test_models(target, selected_colnames):
            with Timer('testing models for [{0}]'.format(target), indent=4):
                with Timer('importing test data', indent=6):
                    x_test, y_test, colnames = import_data(conf['path_to_output'], target, 'test', log_start_time=False, indent=8, level='debug')
                    event_rate = import_object(os.path.join(conf['path_to_output'], 'scoring', target, 'event_rate', 'event_rate.p'), log_start_time=False, indent=8, level='debug')

                if selected_colnames is not None:
                    with Timer('selecting important features', indent=6):
                        x_test, colnames = transform_with_selected_features(x_test, colnames, selected_colnames)

                result = {}

                for model_name in conf['models']:
                    with Timer('importing [{0}] model'.format(model_name), indent=6):
                        model = import_object(os.path.join(conf['path_to_output'], 'models', model_name, target, 'model.p'), log_start_time=False, indent=8, level='debug')

                    with Timer('testing [{0}] model'.format(model_name), indent=6):
                        train_event_rate = get_train_event_rate(conf['train_event_rate'], target)
                        output = test_model(model, x_test, y_test, colnames, conf['verbose'], conf['n_jobs'], conf['eval_metric'], event_rate, train_event_rate, indent=8)

                    with Timer('exporting [{0}] predictions'.format(model_name), indent=6):
                        export_object(output['y_predicted'], os.path.join(conf['path_to_output'], 'predicted', model_name, target, 'test'), 'y_test_predicted.p', log_start_time=False, indent=8, level='debug')

                    result[model_name] = output

            return result

        with Timer('testing models for all targets', indent=2):
            results = {target: test_models(target, selected_colnames_all[target]) for target in conf['targets']}

        with Timer('exporting test results', indent=2):
            export_test_results(results, conf['path_to_output'], conf['models'].keys(), log_start_time=False, indent=4, level='debug')
