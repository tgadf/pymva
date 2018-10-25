from __future__ import division, print_function

from pyspark import SparkContext, HiveContext
from pyspark.sql.types import *
from pyspark.sql import functions as funcs
from pyspark.sql import Window

import sys
import os
import warnings
import functools

import pandas as pd

###############################################################################
# Copied from mtbox/utils.py                                                  #
###############################################################################

import time
import datetime
import re
import traceback
import json
from sklearn.externals.joblib import load


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


def read_colnames(full_filename, **kwargs):
    with Timer('importing [{0}]'.format(full_filename), **kwargs):
        with open(full_filename) as f:
            colnames = [line.strip().upper() for line in f.readlines()]

    return colnames


def import_object(full_filename, **kwargs):
    with Timer('importing [{0}]'.format(full_filename), **kwargs):
        obj = load(full_filename)

    return obj


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
        warnings.warn("The {} function using data on HDFS is deprecated. Please put your scoring data in a Hive table going forward.".format(func.__name__),
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
# Mask logging module                                                         #
###############################################################################


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


def read_score_data(sc, scoring_data_filename_on_hdfs, separator, scoring_data_colnames, num_partitions, **kwargs):
    rdd = sc.textFile(scoring_data_filename_on_hdfs, minPartitions=num_partitions)\
        .map(lambda x: x.split(separator))

    colnames = read_colnames(scoring_data_colnames, **kwargs)

    return rdd, colnames


def read_hive_scoring_data(hc, database_name, table_name, scoring_data_colnames, id_col_name, **kwargs):
    colnames = read_colnames(scoring_data_colnames, **kwargs) + [id_col_name]

    rdd = (hc.read
           .table(database_name + "." + table_name)
           .select(colnames)
           .rdd
           .map(lambda x: [z for z in x]))

    return rdd, colnames


def prepare_scoring(all_colnames, label_decoders, targets, model_names, path_to_output, train_event_rate, fill_values_categorical_missing, **kwargs):
    estimators = []
    feature_info = []
    bias_adjusters = []

    def get_feature_info_for_cat_colname(colname):
        split = colname.split(':')
        return (all_colnames.index(split[0]), label_decoders[split[0]][int(split[1])], fill_values_categorical_missing)

    def get_feature_info_for_num_colname(colname):
        return (all_colnames.index(colname), None, numerical_fill_values[colname])

    for target in targets:
        model = import_object(os.path.join(path_to_output, 'models', model_names[target], target, 'model.p'), **kwargs)
        p = import_object(os.path.join(path_to_output, 'scoring', target, 'event_rate', 'event_rate.p'), **kwargs)
        numerical_fill_values = import_object(os.path.join(path_to_output, 'scoring', target, 'numerical_fill_values', 'numerical_fill_values.p'), **kwargs)
        
        estimator = model['estimator']
        if model['method'] == 'random_forest':
            estimator.n_jobs = 1
        elif model['method'] == 'xgboost':
            estimator.nthread = 1    
        estimators.append(estimator)
        
        colnames = model['colnames']
        feature_info.append([get_feature_info_for_num_colname(colname) if colname.startswith('NUM_') else get_feature_info_for_cat_colname(colname) for colname in colnames])

        # p: event rate, r: train event rate
        # if r is not None, then adjust the score
        r = get_train_event_rate(train_event_rate, target)
        if r:
            bias_adjusters.append((p * (1 - r), r * (1 - p), r - p))
        else:
            bias_adjusters.append((1.0, 1.0, 0.0))

    return estimators, feature_info, bias_adjusters


@deprecated
def score(sc, conf):
    with Timer('[score] command'):
        num_partitions_scoring_data = conf['num_partitions_scoring_data'] if 'num_partitions_scoring_data' in conf else None
        num_partitions_final_scores = conf['num_partitions_final_scores']

        rdd, all_colnames = read_score_data(sc, conf['scoring_data_filename_on_hdfs'], conf['separator'], conf['scoring_data_colnames'], num_partitions_scoring_data, log_start_time=False, indent=2, level='debug')
        rdd.cache()

        label_encoders = import_object(os.path.join(conf['path_to_output'], 'scoring', 'label_encoders', 'label_encoders.p'), log_start_time=False, indent=2, level='debug')
        label_decoders = {cat_colname: {value: key for key, value in label_encoder.items()} for cat_colname, label_encoder in label_encoders.items()}

        best_model_names = pd.read_csv(os.path.join(conf['path_to_output'], 'scoring', 'best_models.csv')).set_index('target')['best_model']

        id_index = all_colnames.index(conf['colname_for_id'].upper())
        targets = conf['targets']

        estimators, feature_info, bias_adjusters = prepare_scoring(all_colnames, label_decoders, targets, best_model_names, conf['path_to_output'], conf['train_event_rate'], conf['fill_values_categorical_missing'], log_start_time=False, indent=2, level='debug')
        
        estimators_bc = sc.broadcast(estimators)
        feature_info_bc = sc.broadcast(feature_info)
        bias_adjusters_bc = sc.broadcast(bias_adjusters)

        def extract_feature(row, index, label, fill_value):
            variable_raw = row[index]

            if variable_raw == u'':
                variable_raw = fill_value

            if label is None:
                variable = float(variable_raw) 
            else:
                variable = 1.0 if variable_raw == label else 0.0

            return variable

        def extract_records(row):
            records = [[extract_feature(row, index, label, fill_value) for index, label, fill_value in feature_info_bc.value[i]] for i in range(len(targets))]
            return (row[id_index], records)

        def score_chunck(chunck):
            ids, records = zip(*chunck)
            records = zip(*records)
            results = [ids]
            for i in range(len(targets)):
                results.append(estimators_bc.value[i].predict_proba(list(records[i]))[:, 1])
            return zip(*results)

        def adjust_bias(row):
            row_adjusted = list(row)
            for i in range(len(row) - 1):
                bias_adjuster = bias_adjusters_bc.value[i]
                row_adjusted[i + 1] = bias_adjuster[0] * row[i + 1] / (bias_adjuster[1] - bias_adjuster[2] * row[i + 1])
            return row_adjusted

        rdd.map(extract_records) \
            .mapPartitions(score_chunck) \
            .map(adjust_bias) \
            .map(lambda cols: ','.join([str(col) for col in cols])) \
            .repartition(num_partitions_final_scores) \
            .saveAsTextFile(conf['path_to_scores_on_hdfs'])


def score_hive(hc, conf):
    with Timer('[score] command'):
        rdd, all_colnames = read_hive_scoring_data(hc, conf['hive_scoring_data']['input_db_name'],
                                                   conf['hive_scoring_data']['input_table_name'],
                                                   os.path.join(conf['path_to_output'], 'scoring', 'colnames_for_scoring.txt'),
                                                   conf['colname_for_id'], log_start_time=False, indent=2, level='debug')
        rdd = rdd.cache()

        label_encoders = import_object(os.path.join(conf['path_to_output'], 'scoring', 'label_encoders', 'label_encoders.p'),
                                       log_start_time=False, indent=2, level='debug')
        label_decoders = {cat_colname: {value: key for key, value in label_encoder.items()} for
                          cat_colname, label_encoder in label_encoders.items()}

        best_model_names = pd.read_csv(os.path.join(conf['path_to_output'], 'scoring', 'best_models.csv')).set_index('target')['best_model']

        id_index = all_colnames.index(conf['colname_for_id'].upper())
        targets = conf['targets']

        estimators, feature_info, bias_adjusters = prepare_scoring(all_colnames, label_decoders, targets,
                                                                   best_model_names, conf['path_to_output'],
                                                                   conf['train_event_rate'],
                                                                   conf['fill_values_categorical_missing'],
                                                                   log_start_time=False, indent=2, level='debug')

        estimators_bc = sc.broadcast(estimators)
        feature_info_bc = sc.broadcast(feature_info)
        bias_adjusters_bc = sc.broadcast(bias_adjusters)

        def extract_feature(row, index, label, fill_value):
            variable_raw = row[index]

            if variable_raw == None or variable_raw == u'':
                variable_raw = fill_value

            if label is None:
                variable = float(variable_raw)
            else:
                variable = 1.0 if variable_raw == label else 0.0

            return variable

        def extract_records(row):
            records = [[extract_feature(row, index, label, fill_value) for index, label, fill_value in feature_info_bc.value[i]] for i in range(len(targets))]
            return (row[id_index], records)

        def score_chunck(chunck):
            ids, records = zip(*chunck)
            records = zip(*records)
            results = [ids]
            for i in range(len(targets)):
                results.append(estimators_bc.value[i].predict_proba(list(records[i]))[:, 1])
            return zip(*results)

        def adjust_bias(row):
            row_adjusted = list(row)
            for i in range(len(row) - 1):
                bias_adjuster = bias_adjusters_bc.value[i]
                row_adjusted[i + 1] = bias_adjuster[0] * row[i + 1] / (bias_adjuster[1] - bias_adjuster[2] * row[i + 1])
            return row_adjusted

        def make_initial_schema(targets):
            schema = StructType()
            schema_str_list = list()

            for header in targets:
                if header.upper() == conf['colname_for_id'].upper():
                    schema.add(header, data_type=StringType())
                    schema_str_list.append(header + " STRING")
                else:
                    schema.add(header + "_SCORE", data_type=FloatType())
                    schema_str_list.append(header + "_SCORE FLOAT")

            return schema, schema_str_list

        headers = [conf['colname_for_id'].upper()] + conf['targets']
        initial_schema, initial_schema_str_list = make_initial_schema(headers)
        score_cols = [col for col in initial_schema.names if col.endswith("_SCORE")]

        def break_ties(target):
            return (funcs.col(target) + (funcs.randn(conf['random_state']) / funcs.lit(10000000000))).alias(target)

        scores_df = (rdd.map(extract_records)
                     .mapPartitions(score_chunck)
                     .map(adjust_bias)
                     .map(lambda x: [z.item() if type(z) is pd.np.float64 else z for z in x])
                     .toDF(initial_schema)
                     .repartition(conf['colname_for_id'])
                     .select(conf['colname_for_id'], *(break_ties(target) for target in score_cols))
                     .cache()
                     )

        if conf['score_analysis_flag'] > 0:
            final_df = None
            for target in score_cols:
                w = Window.orderBy(scores_df[target].desc())
                perc_rank_func = (funcs.percent_rank().over(w))
                rank_func = (funcs.dense_rank().over(w))
                temp = (scores_df.select(conf['colname_for_id'],
                                         target,
                                         rank_func.alias(target + "_rank"),
                                         perc_rank_func.alias(target + "_percent_rank"))
                        .repartition(conf['colname_for_id'])
                        .cache()
                        )
                if final_df is None:
                    final_df = temp
                else:
                    final_df = final_df.join(temp, on=conf['colname_for_id'], how='outer')
        else:
            final_df = scores_df

        full_table_name = conf['hive_scoring_data']['output_db_name'] + '.' + conf['hive_scoring_data']['output_table_name']

        drop_sql = "DROP TABLE IF EXISTS " + full_table_name
        hc.sql(drop_sql)

        final_df.registerTempTable('final_table')
        create_sql = "CREATE TABLE " + full_table_name + " ROW FORMAT DELIMITED FIELDS TERMINATED BY '\\t' STORED AS SEQUENCEFILE AS SELECT * FROM final_table"
        hc.sql(create_sql)

if __name__ == '__main__':
    conf = read_conf_file(sys.argv[1])

    logging = MyLogging()
    setup_logging(conf['verbose'])
    
    sc = SparkContext(appName="mtbox.score")
    hc = HiveContext(sc)

    if "hive_scoring_data" in conf.keys():
        score_hive(hc, conf)
    else:
        score(sc, conf)
