#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:33:36 2017

@author: tgadfort
"""

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

def getRDS(filename):
    readRDS = robjects.r['readRDS']
    df = readRDS('Data1.rds')
    df = pandas2ri.ri2py(df)
    return df

def getRData(filename):
    robjects.r['load'](".RData")