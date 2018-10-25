#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:16:05 2018

@author: tgadfort
"""

import os
import pandas as pd
from yellowbrick.download import download_all

#bikeshare: suitable for regression
#concrete: suitable for regression
#credit: suitable for classification/clustering
#energy: suitable for regression
#game: suitable for classification
#hobbies: suitable for text analysis
#mushroom: suitable for classification/clustering
#occupancy: suitable for classification


## The path to the test data sets
#FIXTURES  = os.path.join(os.getcwd(), "data")
FIXTURES  = "/Users/tgadfort/Documents/pymva/yellowbrick"

## Dataset loading mechanisms
datasets = {
    "bikeshare": os.path.join(FIXTURES, "bikeshare", "bikeshare.csv"),
    "concrete": os.path.join(FIXTURES, "concrete", "concrete.csv"),
    "credit": os.path.join(FIXTURES, "credit", "credit.csv"),
    "energy": os.path.join(FIXTURES, "energy", "energy.csv"),
    "game": os.path.join(FIXTURES, "game", "game.csv"),
    "mushroom": os.path.join(FIXTURES, "mushroom", "mushroom.csv"),
    "occupancy": os.path.join(FIXTURES, "occupancy", "occupancy.csv"),
}


def load_data(name, download=True):
    """
    Loads and wrangles the passed in dataset by name.
    If download is specified, this method will download any missing files.
    """

    # Get the path from the datasets
    path = datasets[name]

    # Check if the data exists, otherwise download or raise
    if not os.path.exists(path):
        if download:
            download_all()
        else:
            raise ValueError((
                "'{}' dataset has not been downloaded, "
                "use the download.py module to fetch datasets"
            ).format(name))


    # Return the data frame
    return pd.read_csv(path)