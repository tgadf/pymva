#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:53:42 2017

@author: tgadfort
"""

from pandas import read_fwf, to_numeric

from logger import info
from colInfo import getNrows, getNcols, isNumeric
from fsio import setSubDir, setFile, isFile
from strops import nice



def dropData(pddf, config):
    info("Dropping columns", ind=4)
    basepath     = config['basepath']
    name         = config['name']
    dropListFile = config['feature']['dropList']
    
    dname  = setSubDir(basepath, ['data', name])
    dlFile = setFile(dname, dropListFile)
    if not isFile(dlFile):
        info("There is no drop file. Not doing anything.", ind=4)
        return

    widths = [int(x) for x in open(dlFile).readline().replace("\n", "").split(',')]
    dlData = read_fwf(dlFile, widths=widths, skiprows=1)
    
    drops  = dlData['Feature'][dlData['Drop?'] == 1]
    info("Dropping "+getNrows(drops, asStr=True)+" columns", ind=6)
    info("Data has "+getNrows(pddf, asStr=True)+" rows and "+getNcols(pddf, asStr=True)+" cols", ind=6)
    pddf.drop(labels=drops.values, axis=1, inplace=True)
    info("Data now has "+getNrows(pddf, asStr=True)+" rows and "+getNcols(pddf, asStr=True)+" cols", ind=6)
    
    

def writeDropList(dlFile, pddf = None, dlData = None):
    info("Writing Drop List", ind=4)
    if dlData is not None:
        maxLen  = max([len(x) for x in dlData['Feature']])
    elif pddf is not None:
        maxLen  = max([len(x) for x in pddf.columns])
        
        
    f = open(dlFile, "w")
    delim=","
    f.write(str(maxLen+3)+delim+str(10)+delim+str(10)+delim+str(10)+delim+str(5)+"\n")
    delim=""
    f.write(nice("Feature",maxLen+3)+delim+nice("Mean",10)+delim+nice("Card",10)+delim+nice("nNA",10)+delim+nice("Drop?",5)+"\n")
    if dlData is not None:
        features = dlData['Feature']
        means    = dlData['Mean']
        cards    = dlData['Card']
        nas      = dlData['nNA']
        drops    = dlData['Drop?']
        for k in range(getNrows(dlData)):
            f.write(nice(features[k], maxLen+3)+delim+nice(str(means[k]),10)+delim+nice(cards[k],10)+delim+nice(str(nas[k]),10)+delim+nice(str(drops[k]),10)+"\n")
    elif pddf is not None:
        for nR,k in enumerate(list(pddf.columns)):
            if nR % 25 == 0: info("Wrote "+str(nR)+"/"+str(len(pddf.columns))+" features.", ind=6)
            na   = sum(pddf[k].isnull())
            if isNumeric(pddf[k]):
                mean = round(pddf[k].mean(), 1)
                f.write(nice(k, maxLen+3)+delim+nice(str(mean),10)+delim+nice("-",10)+delim+nice(str(na),10)+delim+"\n")
            else:
                card = len(pddf[k].unique())
                f.write(nice(k, maxLen+3)+delim+nice("-",10)+delim+nice(str(card),10)+delim+nice(str(na),10)+delim+"\n")
    f.close()
    info("Wrote Drop List to "+dlFile, ind=4)
    


def analyzeColumns(pddf, config):
    info("Analyzing "+getNcols(pddf, asStr=True)+" columns to possible drops.", ind=2)
 
    targetConfig    = config['target']
    targetcol       = targetConfig['colname']
    #problemType     = config['problem']
    #positiveTarget  = config['positiveTarget']
    
    #if isClassification(problemType):
    #    targetData  = trainData[targetcol]
    
    basepath     = config['basepath']
    name         = config['name']
    dropListFile = config['feature']['dropList']
    
    dname  = setSubDir(basepath, ['data', name])
    dlFile = setFile(dname, dropListFile)
    if not isFile(dlFile):
        info("There is no drop file. Not doing anything.", ind=4)
        return
    
    widths = [int(x) for x in open(dlFile).readline().replace("\n", "").split(',')]
    dlData = read_fwf(dlFile, widths=widths, skiprows=1)
    
    
    ## Keep record of overrides
    overrides = dlData['Feature'][dlData['Drop?'].isnull() == False]


    ## Set drop to 0 initially
    dlData['Drop?'].fillna(0, inplace=True)


    ## Drop anything with high cardinality (>50)
    dlData['Card'] = dlData['Card'].apply(to_numeric, errors='coerce')
    dlData['Card'].fillna(0, inplace=True)
    dlData.loc[dlData['Card'] >= 200, 'Drop?'] = 1
    drops  = dlData['Feature'][dlData['Drop?'] == 1]

    
    ## Drop with more than 20% missing data
    maxNA = getNrows(pddf)*0.25
    dlData.loc[dlData['nNA'] >= maxNA, 'Drop?'] = 1
    drops  = dlData['Feature'][dlData['Drop?'] == 1]

    
    ## Fill overrides
    #if getNrows(overrides) > 0:
    #    dlData.loc[dlData['Feature'].isin(overrides['Feature']), 'Drop?'] = overrides['Drop?']
    
    
    ## Lastly, make sure we don't trop the target
    dlData.loc[dlData['Feature'] == targetcol, "Drop?"] = 0
    
    
    ## Show features to drop
    drops  = dlData['Feature'][dlData['Drop?'] == 1]
    print drops


    ## Rewrite drop list
    writeDropList(dlFile, pddf = None, dlData = dlData)