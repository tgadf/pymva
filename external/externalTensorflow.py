#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:44:11 2018

@author: tgadfort
"""


# pip install np_utils
# conda install -c conda-forge tensorflow
# https://www.tensorflow.org/install/

# pip install keras

#import keras
import tensorflow

hello = tensorflow.constant('Hello, TensorFlow!')
sess = tensorflow.Session()
sess.run(hello)
a = tensorflow.constant(10)
b = tensorflow.constant(32)
sess.run(a + b)
sess.close()
