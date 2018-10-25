#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:00:31 2018

@author: tgadfort
"""

# conda install bokeh

def downloadSampleData():
    import bokeh.sampledata
    bokeh.sampledata.download()
    