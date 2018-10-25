#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:20:31 2018

@author: tgadfort
"""

from logger import info
from fsio import setDir, mkDir, isDir

def getDataDir(config):
    name     = config['name']
    datadir  = setDir(config['basepath'], 'data')
    datadir  = setDir(datadir, name)
    if not isDir(datadir):
        info("Making {0}".format(datadir))
        mkDir(datadir)
        
    return datadir


def getPlotsDir(config):
    name     = config['name']
    plotsdir = setDir(config['basepath'], 'plots')
    plotsdir = setDir(plotsdir, name)
    if not isDir(plotsdir):
        info("Making {0}".format(plotsdir))
        mkDir(plotsdir)
        
    return plotsdir


def getModelsDir(config):
    name      = config['name']
    modelsdir = setDir(config['basepath'], 'models')
    modelsdir = setDir(modelsdir, name)
    if not isDir(modelsdir):
        info("Making {0}".format(modelsdir))
        mkDir(modelsdir)
    
    return modelsdir