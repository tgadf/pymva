from __future__ import print_function

from pkg_resources import get_distribution, resource_filename
import logging
import socket

import click
import subprocess32 as subprocess

from .prep import prep
from .explore import explore
from .build import build
from .test import test
from .manifold import manifold
from .utils import commentjson_load, get_open_port


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


def check_version(version_conf):
    version_package = get_distribution('mtbox').version
    if version_conf != version_package:
        raise ValueError('Version in configuration file ({0}) does not match package version ({1}).'.format(version_conf, version_package))


@click.group(chain=True)
@click.version_option()
@click.argument('conf_filename', type=click.Path(exists=True), metavar='<conf_filename>')
@click.pass_context        
def main(ctx, conf_filename):
    """Modeling toolbox."""
    conf = read_conf_file(conf_filename)
    check_version(conf['version'])
    
    ctx.obj = {}
    ctx.obj['conf'] = conf
    ctx.obj['conf_filename'] = conf_filename

    setup_logging(conf['verbose'])


@main.command('prep', short_help='Preprocess input data to prepare train and test data.')
@click.pass_context
def call_prep(ctx):
    conf = ctx.obj['conf']
    prep(conf)


@main.command('explore', short_help='Explore independent variables and compute feature importance.')
@click.pass_context
def call_explore(ctx):
    conf = ctx.obj['conf']
    explore(conf)


@main.command('build', short_help='Build models with train data.')
@click.pass_context
def call_build(ctx):
    conf = ctx.obj['conf']
    build(conf)


@main.command('test', short_help='Test models with test data.')
@click.pass_context
def call_test(ctx):
    conf = ctx.obj['conf']
    test(conf)


@main.command('manifold', short_help='Run manifold learning on train data.')
@click.pass_context
def call_manifold(ctx):
    conf = ctx.obj['conf']
    manifold(conf)


@main.command('prep_spark', short_help='Preprocess input data to prepare train and test data with Spark.')
@click.pass_context
def call_prep_spark(ctx):
    conf = ctx.obj['conf']
    conf_filename = ctx.obj['conf_filename']

    call_spark_application(conf, 'spark/prep_spark.py', [conf_filename])


@main.command('score', short_help='Score new data set with best models.')
@click.pass_context
def call_score(ctx):
    conf = ctx.obj['conf']
    conf_filename = ctx.obj['conf_filename']

    if 'hive_scoring_data' not in conf:
        subprocess.call('hadoop fs -rm -r -f {0}'.format(conf['path_to_scores_on_hdfs']), shell=True)

    call_spark_application(conf, 'spark/score.py', [conf_filename])


def call_spark_application(conf, spark_application_filename, spark_application_arguments):
    enviornment_variables = ''
    if 'enviornment_variables' in conf['pyspark']:
        enviornment_variables = ' '.join(['{0}={1}'.format(key, value) for key, value in conf['pyspark']['enviornment_variables'].iteritems()])
    
    pyspark_bin = conf['pyspark']['pyspark_bin']

    pyspark_script = resource_filename('mtbox', spark_application_filename) + ' '
    pyspark_script += ' '.join(spark_application_arguments)

    options = ''
    if 'options' in conf['pyspark']:
        options = ' '.join(['{0} {1}'.format(key, value) for key, value in conf['pyspark']['options'].iteritems() if key != '--conf'])

        if '--conf' in conf['pyspark']['options']:
            options += ' '
            options += ' '.join(['--conf {0}={1}'.format(key, value) for key, value in conf['pyspark']['options']['--conf'].iteritems()])

    command = ' '.join([enviornment_variables, pyspark_bin, pyspark_script, options])

    subprocess.call(command, shell=True)


@main.command('dashboard', short_help='Launch dashboard server.')
@click.pass_context
def call_dashboard(ctx):
    conf_filename = ctx.obj['conf_filename']

    enviornment_variables = 'MTBOX_CONF_FILENAME={0}'.format(conf_filename)

    # Launch Bokeh server
    address = socket.gethostname()
    bokeh_port = get_open_port()
    
    bokeh_bin = 'bokeh serve'
    bokeh_script = resource_filename('mtbox', 'bokeh/*.py')
    options = '--address {0} --port {1} --host {0}:{1}'.format(address, bokeh_port)

    bokeh_command = ' '.join([enviornment_variables, bokeh_bin, bokeh_script, options])
    bokeh_server = subprocess.Popen(bokeh_command, shell=True)

    # Wait one second
    subprocess.call('sleep 1', shell=True)

    # Launch dashboard with Flask
    flask_script = resource_filename('mtbox', 'dashboard/main.py')
    flask_command = '{0} python {1} {2}'.format(enviornment_variables, flask_script, bokeh_port)
    subprocess.call(flask_command, shell=True)
    
    # Terminate Bokeh server
    bokeh_server.terminate()
