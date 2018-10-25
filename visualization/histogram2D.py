#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:39:33 2018

@author: tgadfort
"""

import numpy as np
import matplotlib.pyplot as plt
 
# Create some random numbers
n = 10000
x = np.random.randn(n)
y = (1.5 * x) + np.random.randn(n)
def plot2DHistogram(x, y, savename = None, showcounts = True):
    # Plot data
    #fig1 = plt.figure()
    #plt.plot(x,y,'.r')
    #plt.xlabel('x')
    #plt.ylabel('y')
     
    # Estimate the 2D histogram
    nbins = 300
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
     
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    fig2 = plt.figure()
    plt.pcolormesh(xedges,yedges,Hmasked,cmap='OrRd')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if showcounts:
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Counts')
    
    if savename is not None:
        print "Saving {0}".format(savename)
        fig2.savefig(savename)
    