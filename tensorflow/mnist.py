#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:58:19 2018

@author: tgadfort
"""
import sys
if '/Users/tgadfort/Python' not in sys.path:
    sys.path.insert(0, '/Users/tgadfort/Python')

from os import stat
from fsio import setDir, setFile, isFile
import gzip
from urllib import urlretrieve
import numpy
from fileio import saveJoblib

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def getMNIST():
    datadir = "/Users/tgadfort/Documents/pymva/data"
    outdir  = setDir(datadir, "mnist")
    
    names = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for name in names:
        url  = "http://yann.lecun.com/exdb/mnist/"+name
        savename = setFile(outdir, name)

        if not isFile(savename):
            urlretrieve(url, savename)
            statinfo = stat(savename)
            print('Succesfully downloaded', savename, statinfo.st_size, 'bytes.')
        
        name = name.replace(".gz", ".p")
        npfile = setFile(outdir, name)
        if not isFile(npfile):
            data = extract_images(savename)
            saveJoblib(npfile, data)