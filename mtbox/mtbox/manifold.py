import os

from sklearn.manifold import TSNE as SK_TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import train_test_split
import numpy as np

from .utils import import_data, transform_with_selected_features, \
    select_initial_features, export_object, Timer, write_colnames


def learn_sk_tsne(x_train, verbose, random_state, params):
    model = SK_TSNE(verbose=verbose, random_state=random_state, n_components=2, **params)

    embedding = model.fit_transform(x_train.toarray())

    return {'embedding': embedding}


def learn_tsne(x_train, verbose, random_state, params):
    model = TSNE(n_jobs=1, **params)

    embedding = model.fit_transform(x_train.toarray())

    return {'embedding': embedding}


def get_manifold_learner(method):
    manifold_learners = {
        'sk_tsne': learn_sk_tsne,
        'tsne': learn_tsne,
    }

    return manifold_learners[method]


def learn_manifold(target, selected_colnames, model_name, conf):
    with Timer('running [{0}] for [{1}]'.format(model_name, target), indent=4):
        with Timer('importing train data', indent=6):
            x_train, y_train, colnames = import_data(conf['path_to_output'], target, 'train', log_start_time=False,
                                               indent=8, level='debug')

        if selected_colnames is not None:
            with Timer('selecting important features', indent=6, level='debug'):
                x_train, colnames = transform_with_selected_features(x_train, colnames, selected_colnames)

        subsample = 'max_num_samples_for_manifold_learning' in conf and y_train.shape[0] > conf['max_num_samples_for_manifold_learning']

        if subsample:
            with Timer('sub-sampling train data', indent=6, level='debug'):
                x_train, _, y_train, _, selected_rows, _ = train_test_split(x_train, y_train, np.arange(y_train.shape[0]),
                                                          train_size=conf['max_num_samples_for_manifold_learning'],
                                                          random_state=conf['random_state'], stratify=y_train)
        else:
            selected_rows = np.arange(y_train.shape[0])

        with Timer('fitting [{0}] for [{1}]'.format(model_name, target), indent=6):
            manifold_learner = get_manifold_learner(conf['manifold_learning'][model_name]['method'])
            output = manifold_learner(x_train, conf['verbose'], conf['random_state'],
                                      conf['manifold_learning'][model_name]['params'])

        with Timer('exporting [{0}] results for [{1}]'.format(model_name, target), indent=6):
            export_object(output['embedding'],
                          os.path.join(conf['path_to_output'], 'embedding', model_name, target),
                          'embedding.p',
                          log_start_time=False, indent=8, level='debug')

            export_object(x_train,
                          os.path.join(conf['path_to_output'], 'embedding', 'data_for_manifold_learning', target),
                          'x_train.p',
                          log_start_time=False, indent=8, level='debug')

            export_object(y_train,
                          os.path.join(conf['path_to_output'], 'embedding', 'data_for_manifold_learning', target),
                          'y_train.p',
                          log_start_time=False, indent=8, level='debug')

            export_object(selected_rows,
                          os.path.join(conf['path_to_output'], 'embedding', 'data_for_manifold_learning', target),
                          'selected_rows.p',
                          log_start_time=False, indent=8, level='debug')

            write_colnames(colnames,
                           os.path.join(conf['path_to_output'], 'embedding', 'data_for_manifold_learning', target),
                           'colnames.txt',
                           log_start_time=False, indent=8, level='debug')


def manifold(conf):
    with Timer('[manifold] command'):
        if 'feature_selection' in conf:
            selected_colnames_all = select_initial_features(conf['path_to_output'], conf['feature_selection'],
                                                            conf['targets'], log_start_time=False, indent=2,
                                                            level='debug')
        else:
            selected_colnames_all = {target: None for target in conf['targets']}

        with Timer('running manifold learning for all targets and models', indent=2):
            Parallel(n_jobs=conf['n_jobs'])(
                delayed(learn_manifold)(target, selected_colnames_all[target], model_name, conf) for target in conf['targets'] for model_name in conf['manifold_learning']
            )