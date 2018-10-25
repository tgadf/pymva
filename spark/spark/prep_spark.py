from __future__ import division, print_function

from pyspark import SparkContext, HiveContext

import sys
import os
import itertools
import operator
import warnings
import functools

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix

###############################################################################
# Copied from mtbox/utils.py                                                  #
###############################################################################

import time
import datetime
import re
import traceback
import json
from sklearn.externals.joblib import dump


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


class JSONLibraryException(Exception):
    ''' This is from commentjson package.
    '''

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
    ''' This is from commentjson package.
    '''
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
    ''' This is from commentjson package.
    '''

    try:
        return commentjson_loads(fp.read(), **kwargs)
    except Exception, e:
        raise JSONLibraryException(e.message)


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


def export_object(obj, path, filename, **kwargs):
    create_folder(path)
    full_filename = os.path.join(path, filename)
    with Timer('exporting [{0}]'.format(full_filename), **kwargs):
        dump(obj, full_filename)


def get_train_event_rate(train_event_rate, target):
    if isinstance(train_event_rate, dict):
        if target in train_event_rate:
            return train_event_rate[target]
        else:
            return train_event_rate['default']
    else:
        return train_event_rate


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("The {} function using data on HDFS is deprecated. Please put your input data in a Hive table going forward.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
        

###############################################################################
# Copied from mtbox/mtbox.py                                                  #
###############################################################################

def read_conf_file(conf_filename):
    with open(conf_filename, "r") as conf_file:    
        conf = commentjson_load(conf_file)
    
    return conf


def setup_logging(verbose):
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    else:
        raise ValueError('Unknown verbose level provided.')

    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=level)

###############################################################################
# Copied from mtbox/prep.py                                                  #
###############################################################################


def read_colnames_for_modeling(path_to_input, targets, **kwargs):
    colnames_for_modeling = list(targets)
    for target in targets:
        colnames_for_modeling.extend(read_colnames(os.path.join(path_to_input, 'colnames_{0}.txt'.format(target)), **kwargs))

    return list(set(colnames_for_modeling))


def select_categorical_colnames(colnames):
    return [colname for colname in colnames if colname.startswith('CAT_')]


def split_and_balance(y, test_size, train_event_rate, random_state):
    y_train, y_test = train_test_split(y, test_size=test_size, random_state=random_state, stratify=y)

    event_rate = y_train.mean()

    if train_event_rate:
        num_non_events = int(round(y_train.sum() * (1 - train_event_rate) / train_event_rate))
        index_train = y_train[y_train < 0.5].sample(num_non_events, random_state=random_state).index.append(y_train[y_train >= 0.5].index)
        y_train = y_train[index_train]

    return y_train, y_test, event_rate


def export_preprocessed_data(path_to_output, target, x_train, y_train, x_test, y_test, event_rate, numerical_fill_values, colnames_with_dummy, **kwargs):
    export_object(x_train, os.path.join(path_to_output, 'preprocessed_data', target, 'train'), 'x_train.p', **kwargs)
    export_object(y_train, os.path.join(path_to_output, 'preprocessed_data', target, 'train'), 'y_train.p', **kwargs)

    export_object(x_test, os.path.join(path_to_output, 'preprocessed_data', target, 'test'), 'x_test.p', **kwargs)
    export_object(y_test, os.path.join(path_to_output, 'preprocessed_data', target, 'test'), 'y_test.p', **kwargs)

    export_object(event_rate, os.path.join(path_to_output, 'scoring', target, 'event_rate'), 'event_rate.p', **kwargs)
    export_object(numerical_fill_values, os.path.join(path_to_output, 'scoring', target, 'numerical_fill_values'), 'numerical_fill_values.p', **kwargs)
    write_colnames(colnames_with_dummy, os.path.join(path_to_output, 'preprocessed_data', target), 'colnames.txt', **kwargs)


###############################################################################
# Mask logging module                                                         #
###############################################################################

import sys

class MyLogging:
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def basicConfig(self, format, datefmt, level):
        self.level = level
        self.format_string = format.replace('%(asctime)s', '{0}').replace('%(message)s', '{1}')
        self.datefmt = datefmt

    def info(self, message):
        if MyLogging.INFO >= self.level:
            self.print_message(message)

    def debug(self, message):
        if MyLogging.DEBUG >= self.level:
            self.print_message(message)

    def print_message(self, message):
        print(self.format_string.format(datetime.datetime.fromtimestamp(time.time()).strftime(self.datefmt), message), file=sys.stderr)

###############################################################################
# Main prep_spark.py module starts from here.                                 #
###############################################################################


def read_input_data(sc, data_filename_on_hdfs, separator, path_to_input, **kwargs):
    rdd = sc.textFile(data_filename_on_hdfs)\
        .zipWithUniqueId() \
        .map(lambda x: (x[1], x[0].split(separator)))

    colnames = read_colnames(os.path.join(path_to_input, 'colnames.txt'), **kwargs)

    return rdd, colnames


def read_hive_table(hc, database_name, table_name, path_to_input, targets, **kwargs):
    table = hc.read.table(database_name + '.' + table_name)

    colnames = [field.name.upper() for field in table.schema.fields]

    colnames_for_modeling = read_colnames_for_modeling(path_to_input, targets, **kwargs)
    usecols = [index for index, colname in enumerate(colnames) if colname in colnames_for_modeling]
    colnames_for_modeling = [colnames[index] for index in usecols]

    rdd = (table
           .select(colnames_for_modeling)
           .rdd
           .zipWithUniqueId()
           .map(lambda x: (x[1], [z for z in x[0]])))

    return rdd, colnames_for_modeling


def collect_target_data(rdd, colnames, targets):
    target_indices = get_indices(targets, colnames)

    collected = rdd \
        .map(lambda x: (x[0], {target_index: int(x[1][target_index]) if x[1][target_index] != u'' else np.nan for target_index in target_indices})) \
        .collect()

    index, data = map(list, zip(*collected))

    df = pd.DataFrame(data, index=index)
    df.columns = [colnames[target_index] for target_index in df.columns.values]

    return df


def collect_target_data_hive(rdd, colnames, targets):
    target_indices = get_indices(targets, colnames)

    collected = rdd \
        .map(lambda x: (x[0], {target_index: int(x[1][target_index]) if x[1][target_index] is not None else np.nan for target_index in target_indices})) \
        .collect()

    index, data = map(list, zip(*collected))

    df = pd.DataFrame(data, index=index)
    df.columns = [colnames[target_index] for target_index in df.columns.values]

    return df


def select_numerical_colnames(colnames):
    return [colname for colname in colnames if colname.startswith('NUM_')]


def get_indices(selected_colnames, colnames):
    return [colnames.index(selcted_colname) for selcted_colname in selected_colnames]


def prepare_label_encoders(rdd, colnames, path_to_input, targets, fill_values_categorical_missing, **kwargs):
    colnames_for_modeling = read_colnames_for_modeling(path_to_input, targets, **kwargs)
    cat_colnames = select_categorical_colnames(colnames_for_modeling)

    label_encoders = {}

    if len(cat_colnames) > 0:
        cat_indices = get_indices(cat_colnames, colnames)

        distinct_values = rdd \
            .flatMap(lambda x: [(cat_index, x[1][cat_index]) for cat_index in cat_indices]) \
            .map(lambda x: (x[0], fill_values_categorical_missing if x[1] == u'' else x[1])) \
            .distinct() \
            .collect()

        label_encoders = {colnames[cat_index]: {item[1]:index for index, item in enumerate(subiter)} for cat_index, subiter in itertools.groupby(sorted(distinct_values), operator.itemgetter(0))}

    return label_encoders


def prepare_label_encoders_hive(rdd, colnames, path_to_input, targets, fill_values_categorical_missing, **kwargs):
    colnames_for_modeling = read_colnames_for_modeling(path_to_input, targets, **kwargs)
    cat_colnames = select_categorical_colnames(colnames_for_modeling)

    label_encoders = {}

    if len(cat_colnames) > 0:
        cat_indices = get_indices(cat_colnames, colnames)

        distinct_values = rdd \
            .flatMap(lambda x: [(cat_index, x[1][cat_index]) for cat_index in cat_indices]) \
            .map(lambda x: (x[0], fill_values_categorical_missing if x[1] is u'' else x[1])) \
            .distinct() \
            .collect()

        label_encoders = {colnames[cat_index]: {item[1]:index for index, item in enumerate(subiter)} for cat_index, subiter in itertools.groupby(sorted(distinct_values), operator.itemgetter(0))}

    return label_encoders


def get_start_indices(cat_colnames, label_encoders):
    cat_start_indices = []
    num_start_index = 0
    
    if len(cat_colnames) > 0:
        start_indices = np.array([len(label_encoders[cat_colname]) for cat_colname in cat_colnames]).cumsum()
        cat_start_indices = [0] + start_indices[:-1].tolist()
        num_start_index = start_indices[-1]
    
    return cat_start_indices, num_start_index


def fill_missing_values(data, column_indices, n_rows, colnames_with_dummy, fill_values=None):
    df = pd.DataFrame({'data': data, 'column_indices': column_indices})
    if fill_values is None:
        fill_values_temp = df.groupby('column_indices')['data'].sum() / n_rows
        fill_values = pd.Series(0, index=colnames_with_dummy)
        fill_values[[colnames_with_dummy[index] for index in fill_values_temp.index.values]] = fill_values_temp
    grouped = df.groupby('column_indices')['data']
    data_filled = pd.concat(group.fillna(fill_values[colnames_with_dummy[index]]) for index, group in grouped).loc[df.index].values

    return data_filled, fill_values


def collect_input_data(sc, rdd, y, colnames, cat_colnames, cat_indices, num_indices, colnames_with_dummy, label_encoders, fill_values_categorical_missing, fill_values=None):
    rdd_indices_bc = sc.broadcast({rdd_index: key for key, rdd_index in enumerate(y.index.values)})
    label_encoders_bc = sc.broadcast(label_encoders)

    cat_start_indices, num_start_index = get_start_indices(cat_colnames, label_encoders)

    def extract_sparse_components(x):
        row = x[1]

        cat_data = num_data = []
        cat_column_indices = num_column_indices = []

        for cat_index in cat_indices:
            if row[cat_index] == u'':
                row[cat_index] = label_encoders_bc.value[colnames[cat_index]][fill_values_categorical_missing]
            else:
                row[cat_index] = label_encoders_bc.value[colnames[cat_index]][row[cat_index]]

        for num_index in num_indices:
            if row[num_index] == u'':
                row[num_index] = np.nan
            else:
                row[num_index] = float(row[num_index])

        cat_column_indices = [row[cat_index] + cat_start_index for cat_index, cat_start_index in zip(cat_indices, cat_start_indices)]
        cat_data = [1] * len(cat_column_indices)

        num_data_and_column_indices = [(row[num_index], num_column_index + num_start_index) for num_column_index, num_index in enumerate(num_indices) if row[num_index] != 0]
        if len(num_data_and_column_indices) > 0:
            num_data, num_column_indices = map(list, zip(*num_data_and_column_indices))

        data = cat_data + num_data
        column_indices = cat_column_indices + num_column_indices
        
        return (rdd_indices_bc.value[x[0]], (data, column_indices))

    collected = rdd \
        .filter(lambda x: x[0] in rdd_indices_bc.value) \
        .map(extract_sparse_components) \
        .sortByKey() \
        .map(lambda (k, v): (k, v[0], v[1])) \
        .collect()

    keys, data_collected, column_indices_collected = zip(*collected)
    
    if not all(index == key for index, key in enumerate(keys)):
        raise ValueError('PySpark did not return results in order.')

    data = list(itertools.chain.from_iterable(data_collected))
    column_indices = list(itertools.chain.from_iterable(column_indices_collected))

    row_indices_collected = ([row_index] * len(d) for row_index, d in enumerate(data_collected))
    row_indices = list(itertools.chain.from_iterable(row_indices_collected))

    n_rows = len(collected)
    n_cols = len(colnames_with_dummy)

    data_filled, fill_values = fill_missing_values(data, column_indices, n_rows, colnames_with_dummy, fill_values=fill_values)

    x = csr_matrix((data_filled, (row_indices, column_indices)), shape=(n_rows, n_cols))

    return x, fill_values


def collect_input_data_hive(sc, rdd, y, colnames, cat_colnames, cat_indices, num_indices, colnames_with_dummy,
                       label_encoders, fill_values_categorical_missing, fill_values=None):
    rdd_indices_bc = sc.broadcast({rdd_index: key for key, rdd_index in enumerate(y.index.values)})
    label_encoders_bc = sc.broadcast(label_encoders)

    cat_start_indices, num_start_index = get_start_indices(cat_colnames, label_encoders)

    def extract_sparse_components(x):
        row = x[1]

        num_data = []
        num_column_indices = []

        for cat_index in cat_indices:
            if row[cat_index] == u'':
                row[cat_index] = label_encoders_bc.value[colnames[cat_index]][fill_values_categorical_missing]
            else:
                row[cat_index] = label_encoders_bc.value[colnames[cat_index]][row[cat_index]]

        for num_index in num_indices:
            if row[num_index] == None:
                row[num_index] = np.nan
            else:
                row[num_index] = float(row[num_index])

        cat_column_indices = [row[cat_index] + cat_start_index for cat_index, cat_start_index in
                              zip(cat_indices, cat_start_indices)]
        cat_data = [1] * len(cat_column_indices)

        num_data_and_column_indices = [(row[num_index], num_column_index + num_start_index) for
                                       num_column_index, num_index in enumerate(num_indices) if row[num_index] != 0]
        if len(num_data_and_column_indices) > 0:
            num_data, num_column_indices = map(list, zip(*num_data_and_column_indices))

        data = cat_data + num_data
        column_indices = cat_column_indices + num_column_indices
        return (rdd_indices_bc.value[x[0]], (data, column_indices))

    collected = rdd \
        .filter(lambda x: x[0] in rdd_indices_bc.value) \
        .map(extract_sparse_components) \
        .sortByKey() \
        .map(lambda (k, v): (k, v[0], v[1])) \
        .collect()

    keys, data_collected, column_indices_collected = zip(*collected)

    if not all(index == key for index, key in enumerate(keys)):
        raise ValueError('PySpark did not return results in order.')

    data = list(itertools.chain.from_iterable(data_collected))
    column_indices = list(itertools.chain.from_iterable(column_indices_collected))

    row_indices_collected = ([row_index] * len(d) for row_index, d in enumerate(data_collected))
    row_indices = list(itertools.chain.from_iterable(row_indices_collected))

    n_rows = len(collected)
    n_cols = len(colnames_with_dummy)

    data_filled, fill_values = fill_missing_values(data, column_indices, n_rows, colnames_with_dummy,
                                                   fill_values=fill_values)

    x = csr_matrix((data_filled, (row_indices, column_indices)), shape=(n_rows, n_cols))

    return x, fill_values


def get_colnames_with_dummy(cat_colnames, num_colnames, label_encoders):
    colnames_with_dummy = num_colnames

    if len(cat_colnames) > 0:
        n_values = [len(label_encoders[cat_colname]) for cat_colname in cat_colnames]

        colnames_with_dummy = []
        for cat_column, n_value in zip(cat_colnames, n_values):
            colnames_with_dummy += ['{0}:{1}'.format(cat_column, i) for i in range(n_value)]
        colnames_with_dummy += num_colnames

    return colnames_with_dummy


@deprecated
def prep(sc, conf):
    with Timer('[prep_spark] command'):
        rdd, all_colnames = read_input_data(sc, conf['data_filename_on_hdfs'], conf['separator'], conf['path_to_input'], log_start_time=False, indent=2, level='debug')
        rdd.cache()

        with Timer('preparing label encoders for categorical features', indent=2):
            label_encoders = prepare_label_encoders(rdd, all_colnames, conf['path_to_input'], conf['targets'], conf['fill_values_categorical_missing'], log_start_time=False, indent=4, level='debug')

        with Timer('collecting target variables', indent=2):
            df_targets = collect_target_data(rdd, all_colnames, conf['targets'])

        def preprocess_input_data(target):
            with Timer('preprocessing input data for [{0}]'.format(target), indent=4):
                colnames = read_colnames(os.path.join(conf['path_to_input'], 'colnames_{0}.txt'.format(target)), log_start_time=False, indent=6, level='debug')

                cat_colnames = select_categorical_colnames(colnames)
                num_colnames = select_numerical_colnames(colnames)

                cat_indices = get_indices(cat_colnames, all_colnames)
                num_indices = get_indices(num_colnames, all_colnames)

                colnames_with_dummy = get_colnames_with_dummy(cat_colnames, num_colnames, label_encoders)

                with Timer('splitting train and test data', indent=6):
                    train_event_rate = get_train_event_rate(conf['train_event_rate'], target)
                    y_train, y_test, event_rate = split_and_balance(df_targets[target].dropna(), conf['test_size'], train_event_rate, conf['random_state'])

                with Timer('collecting train data', indent=6):
                    x_train, fill_values = collect_input_data(sc, rdd, y_train, all_colnames, cat_colnames, cat_indices, num_indices, colnames_with_dummy, label_encoders, conf['fill_values_categorical_missing'])
                
                with Timer('collecting test data', indent=6):
                    x_test, _ = collect_input_data(sc, rdd, y_test, all_colnames, cat_colnames, cat_indices, num_indices, colnames_with_dummy, label_encoders, conf['fill_values_categorical_missing'], fill_values=fill_values)

                with Timer('exporting preprocessed data', indent=6):
                    export_preprocessed_data(conf['path_to_output'], target, x_train, y_train, x_test, y_test, event_rate, fill_values, colnames_with_dummy, log_start_time=False, indent=8, level='debug')
       
        with Timer('preprocessing input data for all targets', indent=2):
            map(preprocess_input_data, conf['targets'])

        export_object(label_encoders, os.path.join(conf['path_to_output'], 'scoring', 'label_encoders'), 'label_encoders.p', log_start_time=False, indent=2, level='debug')


def prep_hive(hc, conf):
    with Timer('[prep_spark] command'):
        rdd, all_colnames = read_hive_table(hc, conf['hive_input_data']['db_name'], conf['hive_input_data']['table_name'],
                              conf['path_to_input'], conf['targets'], log_start_time=False, indent=2, level='debug')

        rdd = rdd.cache()

        with Timer('preparing label encoders for categorical features', indent=2):
            label_encoders = prepare_label_encoders_hive(rdd, all_colnames, conf['path_to_input'], conf['targets'],
                                                    conf['fill_values_categorical_missing'], log_start_time=False, indent=4,
                                                    level='debug')

        with Timer('collecting target variables', indent=2):
            df_targets = collect_target_data_hive(rdd, all_colnames, conf['targets'])

        def preprocess_input_data(target):
            with Timer('preprocessing input data for [{0}]'.format(target), indent=4):
                colnames = read_colnames(os.path.join(conf['path_to_input'], 'colnames_{0}.txt'.format(target)),
                                         log_start_time=False, indent=6, level='debug')

                cat_colnames = select_categorical_colnames(colnames)
                num_colnames = select_numerical_colnames(colnames)

                cat_indices = get_indices(cat_colnames, all_colnames)
                num_indices = get_indices(num_colnames, all_colnames)

                colnames_with_dummy = get_colnames_with_dummy(cat_colnames, num_colnames, label_encoders)

                with Timer('splitting train and test data', indent=6):
                    train_event_rate = get_train_event_rate(conf['train_event_rate'], target)
                    y_train, y_test, event_rate = split_and_balance(df_targets[target].dropna(), conf['test_size'],
                                                                    train_event_rate, conf['random_state'])

                with Timer('collecting train data', indent=6):
                    x_train, fill_values = collect_input_data_hive(sc, rdd, y_train, all_colnames, cat_colnames, cat_indices,
                                                              num_indices, colnames_with_dummy, label_encoders,
                                                              conf['fill_values_categorical_missing'])

                with Timer('collecting test data', indent=6):
                    x_test, _ = collect_input_data_hive(sc, rdd, y_test, all_colnames, cat_colnames, cat_indices, num_indices,
                                                   colnames_with_dummy, label_encoders,
                                                   conf['fill_values_categorical_missing'], fill_values=fill_values)

                with Timer('exporting preprocessed data', indent=6):
                    export_preprocessed_data(conf['path_to_output'], target, x_train, y_train, x_test, y_test, event_rate,
                                             fill_values, colnames_with_dummy, log_start_time=False, indent=8,
                                             level='debug')

        with Timer('preprocessing input data for all targets', indent=2):
            map(preprocess_input_data, conf['targets'])

        export_object(label_encoders, os.path.join(conf['path_to_output'], 'scoring', 'label_encoders'), 'label_encoders.p',
                      log_start_time=False, indent=2, level='debug')


def suppress_spark_messages(sc):
    spark_logger = sc._jvm.org.apache.log4j
    spark_logger.LogManager.getLogger("org").setLevel( spark_logger.Level.ERROR )
    spark_logger.LogManager.getLogger("akka").setLevel( spark_logger.Level.ERROR )


if __name__ == '__main__':
    conf = read_conf_file(sys.argv[1])

    logging = MyLogging()
    setup_logging(conf['verbose'])
    
    sc = SparkContext(appName="mtbox.prep_spark")
    hc = HiveContext(sc)

    if conf['verbose'] <= 1:
        suppress_spark_messages(sc)
    if "hive_input_data" in conf.keys():
        prep_hive(hc, conf)
    else:
        prep(sc, conf)
