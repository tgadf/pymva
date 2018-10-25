#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:43:10 2018

@author: tgadfort
"""

import logging
import inspect

logger=logging.getLogger(__name__)


def setupLogger():
    logging.basicConfig(level=logging.INFO,
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s')


def getInd(ind = 0):
    try:
        frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
                inspect.currentframe())[1]
        line=lines[0]
        indentation_level=line.find(line.lstrip()) + ind
    except:
        indentation_level = ind
    return indentation_level


def info(msg, ind = 0):
    indentation_level = getInd(ind)
    logger.info('{i} [{m}]'.format(i='.'*indentation_level, m=msg))


def debug(msg, ind = 0):
    indentation_level = getInd(ind)
    logger.debug('{i} [{m}]'.format(i='.'*indentation_level, m=msg))


def error(msg, ind = 0):
    indentation_level = getInd(ind)
    logger.error('{i} [{m}]'.format(i='.'*indentation_level, m=msg))
