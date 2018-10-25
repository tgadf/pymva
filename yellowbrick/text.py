#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:13:47 2018

@author: tgadfort
"""

from yellowbrick.text import FreqDistVisualizer
from yellowbrick.text import TSNEVisualizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from corpus import load_corpus

#from yellowbrick.text.freqdist import FreqDistVisualizer


def showToken():
    
    corpus = load_corpus("hobbies")
    
    vectorizer = CountVectorizer()
    docs       = vectorizer.fit_transform(corpus.data)
    features   = vectorizer.get_feature_names()
    
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    visualizer.poof()
    
    
    vectorizer = CountVectorizer(stop_words='english')
    docs       = vectorizer.fit_transform(corpus.data)
    features   = vectorizer.get_feature_names()
    
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    visualizer.poof()
    
    
    from collections import defaultdict
    
    hobbies = defaultdict(list)
    for text, label in zip(corpus.data, corpus.categories):
        hobbies[label].append(text)
    vectorizer = CountVectorizer(stop_words='english')
    docs       = vectorizer.fit_transform(text for text in hobbies['cooking'])
    features   = vectorizer.get_feature_names()
    
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    visualizer.poof()
    
    
def showTSNE():
    # Load the data and create document vectors
    corpus = load_corpus('hobbies')
    tfidf  = TfidfVectorizer()

    docs   = tfidf.fit_transform(corpus.data)
    labels = corpus.target
    
    # Create the visualizer and draw the vectors
    tsne = TSNEVisualizer()
    tsne.fit(docs, labels)
    tsne.poof()
    
    
    # Only visualize the sports, cinema, and gaming classes
    tsne = TSNEVisualizer(classes=['sports', 'cinema', 'gaming'])
    tsne.fit(docs, labels)
    tsne.poof()
    
    
    # Apply clustering instead of class names.
    from sklearn.cluster import KMeans

    clusters = KMeans(n_clusters=5)
    clusters.fit(docs)

    tsne = TSNEVisualizer()
    tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
    tsne.poof()