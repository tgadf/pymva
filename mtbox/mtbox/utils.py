import os
import re
import traceback
import json
import time
import datetime
import logging
import socket

import pandas as pd
from sklearn.externals.joblib import dump, load
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

###############################################################################
# Copied from commentjson package                                             #
###############################################################################

class JSONLibraryException(Exception):
    def __init__(self, json_error=""):
        tb = traceback.format_exc()
        tb = '\n'.join(' ' * 4 + line_ for line_ in tb.split('\n'))
        message = [
            'JSON Library Exception\n',
            ('Exception thrown by JSON library (json): '
             '\033[4;37m%s\033[0m\n' % json_error),
            '%s' % tb,
        ]
        Exception.__init__(self, '\n'.join(message))


def commentjson_loads(text, **kwargs):
    regex = r'\s*(#|\/{2}).*$'
    regex_inline = r'(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$'
    lines = text.split('\n')

    for index, line in enumerate(lines):
        if re.search(regex, line):
            if re.search(r'^' + regex, line, re.IGNORECASE):
                lines[index] = ""
            elif re.search(regex_inline, line):
                lines[index] = re.sub(regex_inline, r'\1', line)

    try:
        return json.loads('\n'.join(lines), **kwargs)
    except Exception, e:
        raise JSONLibraryException(e.message)


def commentjson_load(fp, **kwargs):
    try:
        return commentjson_loads(fp.read(), **kwargs)
    except Exception, e:
        raise JSONLibraryException(e.message)

###############################################################################

class Timer:    
    def __init__(self, message='', log_start_time=True, indent=0, level='info'):
        self.message = message
        self.indent = '.' * indent
        self.log_start_time = log_start_time

        if level == 'info':
            self.logger = logging.info
        elif level == 'debug':
            self.logger = logging.debug

    def __enter__(self):
        self.start = time.time()

        if self.message != '' and self.log_start_time:
            self.logger('{0}Started {1}.'.format(self.indent, self.message))

        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

        if self.message != '':
            message = 'Completed {0}. '.format(self.message)
        else:
            message = self.message

        self.logger('{0}{1}(Elapsed time: {2})'.format(self.indent, message, datetime.timedelta(seconds=round(self.interval))))


def create_folder(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def read_colnames(full_filename, **kwargs):
    with Timer('importing [{0}]'.format(full_filename), **kwargs):
        with open(full_filename) as f:
            colnames = [line.strip().upper() for line in f.readlines()]

    return colnames


def write_colnames(colnames, path, filename, **kwargs):
    create_folder(path)
    full_filename = os.path.join(path, filename)
    with Timer('exporting [{0}]'.format(full_filename), **kwargs):
        with open(full_filename, 'w') as f:
            f.write('\n'.join(colnames))


def write_dataframe(df, path, filename, **kwargs):
    create_folder(path)
    full_filename = os.path.join(path, filename)
    with Timer('exporting [{0}]'.format(full_filename), **kwargs):
        df.to_csv(full_filename, index=False)


def import_object(full_filename, **kwargs):
    with Timer('importing [{0}]'.format(full_filename), **kwargs):
        obj = load(full_filename)

    return obj


def export_object(obj, path, filename, **kwargs):
    create_folder(path)
    full_filename = os.path.join(path, filename)
    with Timer('exporting [{0}]'.format(full_filename), **kwargs):
        dump(obj, full_filename)


def generate_feature_importance_report(colnames, feature_importance):
    df = pd.DataFrame({'colname': colnames, 'feature_importance': feature_importance})
    df.insert(0, 'rank', df['feature_importance'].rank(method='min', ascending=False).astype(int))
    df.sort_values('rank', inplace=True)
    df['cumulative_feature_importance'] = df['feature_importance'].cumsum()

    return df


def transform_with_selected_features(x_train, colnames, selected_colnames):
    indices = [colnames.index(selected_colname) for selected_colname in selected_colnames]

    x_train = x_train.tocsc()[:,indices].tocsr()
    colnames = selected_colnames

    return x_train, colnames


def import_data(path_to_output, target, role, **kwargs):
    X = import_object(os.path.join(path_to_output, 'preprocessed_data', target, role, 'x_{0}.p'.format(role)), **kwargs)
    y = import_object(os.path.join(path_to_output, 'preprocessed_data', target, role, 'y_{0}.p'.format(role)), **kwargs)

    colnames = read_colnames(os.path.join(path_to_output, 'preprocessed_data', target, 'colnames.txt'), **kwargs)

    return X, y, colnames


def select_top(df, threshold):
    grouped = df[((df['rank'] <= threshold) & (df['rank'] != -1)) | (df['rank'] == 0)].groupby('target')

    # To make sure # of selected variables is not greater than threshold when there are ties.
    df_1 = grouped.filter(lambda x: len(x) <= threshold)
    df_2 = grouped.filter(lambda x: len(x) > threshold) \
        .groupby('target') \
        .apply(lambda x: x[x['rank'] < x['rank'].max()]) \
        .reset_index(drop=True)

    return pd.concat([df_1, df_2], ignore_index=True)


def select_at_least(df, threshold):
    return df[((df['feature_importance'] >= threshold) & (df['rank'] != -1)) | (df['rank'] == 0)]


def select_coverage(df, threshold):
    return df[((df['cumulative_feature_importance'] <= threshold) & (df['rank'] != -1)) | (df['rank'] == 0)]


def get_feature_selector(method):
    feature_selectors = {
        'top': select_top,
        'at_least': select_at_least,
        'coverage': select_coverage 
    }

    return feature_selectors[method]


def select_initial_features(path_to_output, params, targets, **kwargs):
    full_filename = os.path.join(path_to_output, 'feature_importance', 'feature_importance_{0}.csv'.format(params['feature_importance']))

    with Timer('importing [{0}]'.format(full_filename), **kwargs):
        df = pd.read_csv(full_filename)

    feature_selector = get_feature_selector(params['which'])

    important_features = feature_selector(df, params['threshold'])

    return {target: important_features.loc[important_features['target'] == target, 'colname'].values.tolist() for target in targets}


def get_scorer(eval_metric):
    scorers = {
        'roc_auc': roc_auc_score,
        'average_precision': average_precision_score,
        'log_loss': log_loss
    }

    return scorers[eval_metric]


def append_target_column(df_base, target):
    df = df_base.copy()
    df.insert(0, 'target', target)
    
    return df


def append_model_name_to_colnames(df, model_name):
    new_colnames = ['target']
    new_colnames.extend(['{0}.{1}'.format(model_name, column) for column in df.columns.values.tolist()[1:]])
    df.columns = new_colnames
    
    return df


def collect_item_in_results(results, model_name, item, should_append_model_name_to_colnames=True):
    df = pd.concat((append_target_column(results[target][model_name][item], target) for target in results), ignore_index=True)

    if should_append_model_name_to_colnames:
        df = append_model_name_to_colnames(df, model_name)
    
    return df


def concat_reports_horizontally(dfs):
    df_concat = pd.concat((df.set_index('target') for df in dfs), axis=1)
    df_concat.reset_index(inplace=True)
    return df_concat


def get_train_event_rate(train_event_rate, target):
    if isinstance(train_event_rate, dict):
        if target in train_event_rate:
            return train_event_rate[target]
        else:
            return train_event_rate['default']
    else:
        return train_event_rate


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port
