import os

from numpy import float64 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.externals.joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from .utils import read_colnames, write_colnames, export_object, Timer, get_train_event_rate


def read_colnames_for_modeling(path_to_input, targets, **kwargs):
    colnames_for_modeling = list(targets)
    for target in targets:
        colnames_for_modeling.extend(read_colnames(os.path.join(path_to_input, 'colnames_{0}.txt'.format(target)), **kwargs))

    return list(set(colnames_for_modeling))


def read_input_data(path_to_input, data_filename, separator, targets, **kwargs):
    colnames = read_colnames(os.path.join(path_to_input, 'colnames.txt'), log_start_time=False, **kwargs)
    colnames_for_modeling = read_colnames_for_modeling(path_to_input, targets, log_start_time=False, **kwargs)

    usecols = [index for index, colname in enumerate(colnames) if colname in colnames_for_modeling]
    colnames_for_modeling = [colnames[index] for index in usecols]

    dtype = {colname: str if colname.startswith('CAT_') else float64 for colname in colnames_for_modeling}

    full_filename = os.path.join(path_to_input, data_filename)
    with Timer('importing [{0}]'.format(full_filename), **kwargs):
        df = pd.read_csv(full_filename, usecols=usecols, names=colnames_for_modeling, dtype=dtype, sep=separator)

    return df

def select_categorical_colnames(colnames):
    return [colname for colname in colnames if colname.startswith('CAT_')]


def fill_missing_values(df, fill_method):
    fill_values = fill_method

    # TODO make it Python 3 compatible
    if isinstance(fill_method, basestring):
        if fill_method == 'mean':
            fill_values = df.mean()
        # elif fill_method == 'median':
        #     fill_values = df.median()
        # elif fill_method == 'mode':
        #     fill_values = df.mode().iloc[0]

    return df.fillna(fill_values), fill_values


def encode_categorical_column(cat_column, fill_method):
    cat_column, _ = fill_missing_values(cat_column, fill_method)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(cat_column)

    return cat_column.name, label_encoder, encoded


def encode_categorical_columns(df, n_jobs, fill_method):
    cat_colnames = select_categorical_colnames(df.columns.tolist())

    df_num = df
    label_encoders = {}

    if len(cat_colnames) > 0:
        results = Parallel(n_jobs=n_jobs)(delayed(encode_categorical_column)(df[cat_colname], fill_method) for cat_colname in cat_colnames)

        label_encoders = {cat_colname: label_encoder for cat_colname, label_encoder, encoded in results}

        df_cat = pd.DataFrame({cat_colname: encoded for cat_colname, label_encoder, encoded in results})

        df.drop(cat_colnames, axis=1, inplace=True)

        df_num = df_cat.join(df)

    return df_num, label_encoders


def split_and_balance(y, test_size, train_event_rate, random_state):
    y_train, y_test = train_test_split(y, test_size=test_size, random_state=random_state, stratify=y)
    print "len(train):    ", len(y_train)
    print "len(test):     ", len(y_test)

    event_rate = y_train.mean()
    print "event rate:    ", event_rate
    print "train rate:    ", train_event_rate

    if train_event_rate:
        num_non_events = int(round(y_train.sum() * (1 - train_event_rate) / train_event_rate))
        print "num_non_events:",num_non_events
        index_train = y_train[y_train < 0.5].sample(num_non_events).index.append(y_train[y_train >= 0.5].index)
        #index_train = y_train[y_train < 0.5].sample(num_non_events, random_state=random_state).index.append(y_train[y_train >= 0.5].index)
        y_train = y_train[index_train]

    print "len(all):      ", len(y)
    print "len(train):    ", len(y_train)
    print "len(test):     ", len(y_test)

    return y_train, y_test, event_rate


def prepare_one_hot_encoding(colnames, label_encoders):
    cat_colnames = select_categorical_colnames(colnames)

    one_hot_encoder = None
    colnames_with_dummy = colnames

    if len(cat_colnames) > 0:
        n_values = [len(label_encoders[cat_colname].classes_) for cat_colname in cat_colnames]
        categorical_features = [colnames.index(cat_colname) for cat_colname in cat_colnames]

        one_hot_encoder = OneHotEncoder(n_values=n_values, categorical_features=categorical_features)

        colnames_with_dummy = []
        for column_cat, n_value in zip(cat_colnames, n_values):
            colnames_with_dummy += ['{0}:{1}'.format(column_cat, i) for i in range(n_value)]
        colnames_with_dummy += [colname for colname in colnames if not colname.startswith('CAT_')]

    return one_hot_encoder, colnames_with_dummy


def export_preprocessed_data(path_to_output, target, x_train, y_train, x_test, y_test, event_rate, numerical_fill_values, colnames_with_dummy, **kwargs):
    export_object(x_train, os.path.join(path_to_output, 'preprocessed_data', target, 'train'), 'x_train.p', **kwargs)
    export_object(y_train, os.path.join(path_to_output, 'preprocessed_data', target, 'train'), 'y_train.p', **kwargs)

    export_object(x_test, os.path.join(path_to_output, 'preprocessed_data', target, 'test'), 'x_test.p', **kwargs)
    export_object(y_test, os.path.join(path_to_output, 'preprocessed_data', target, 'test'), 'y_test.p', **kwargs)

    export_object(event_rate, os.path.join(path_to_output, 'scoring', target, 'event_rate'), 'event_rate.p', **kwargs)
    export_object(numerical_fill_values, os.path.join(path_to_output, 'scoring', target, 'numerical_fill_values'), 'numerical_fill_values.p', **kwargs)
    write_colnames(colnames_with_dummy, os.path.join(path_to_output, 'preprocessed_data', target), 'colnames.txt', **kwargs)


def label_encoder_to_dict(label_encoder):
    return {label: index for index, label in enumerate(label_encoder.classes_)}


def prep(conf):
    with Timer('[prep] command'):
        with Timer('reading input data', indent=2):
            df = read_input_data(conf['path_to_input'], conf['data_filename'], conf['separator'], conf['targets'], indent=2, level='debug')

        with Timer('encoding categorical features', indent=2):
            df, label_encoders = encode_categorical_columns(df, conf['n_jobs'], conf['fill_values_categorical_missing'])

        def preprocess_input_data(target):
            with Timer('preprocessing input data for [{0}]'.format(target), indent=4):
                colnames = read_colnames(os.path.join(conf['path_to_input'], 'colnames_{0}.txt'.format(target)), log_start_time=False, indent=6, level='debug')

                with Timer('splitting train and test data', indent=6):
                    train_event_rate = get_train_event_rate(conf['train_event_rate'], target)
                    print train_event_rate
                    y_train, y_test, event_rate = split_and_balance(df[target].dropna(), conf['test_size'], train_event_rate, conf['random_state'])

                with Timer('collecting train data', indent=6):
                    x_train, numerical_fill_values = fill_missing_values(df.loc[y_train.index, colnames], 'mean')
                
                with Timer('collecting test data', indent=6):
                    x_test, _ = fill_missing_values(df.loc[y_test.index, colnames], numerical_fill_values)

                one_hot_encoder, colnames_with_dummy = prepare_one_hot_encoding(colnames, label_encoders)

                if one_hot_encoder is not None:
                    with Timer('one-hot encoding categorical features and conversion to sparse format', indent=6):
                        x_train = one_hot_encoder.fit_transform(x_train).tocsr()
                        x_test = one_hot_encoder.transform(x_test).tocsr()
                else:
                    with Timer('conversion to sparse format', indent=6):
                        x_train = csr_matrix(x_train)
                        x_test = csr_matrix(x_test)

                with Timer('exporting preprocessed data', indent=6):
                    export_preprocessed_data(conf['path_to_output'], target, x_train, y_train, x_test, y_test, event_rate, numerical_fill_values, colnames_with_dummy, log_start_time=False, indent=8, level='debug')
            
        with Timer('preprocessing input data for all targets', indent=2):
            map(preprocess_input_data, conf['targets'])

        export_object({cat_colname: label_encoder_to_dict(label_encoder) for cat_colname, label_encoder in label_encoders.iteritems()}, os.path.join(conf['path_to_output'], 'scoring', 'label_encoders'), 'label_encoders.p', log_start_time=False, indent=2, level='debug')
