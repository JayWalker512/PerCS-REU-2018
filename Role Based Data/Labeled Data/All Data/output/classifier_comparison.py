#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:45:46 2018

@author: jaywalker
"""
from __future__ import print_function
from __future__ import division
import sys
import csv
import io
import math

import numpy
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

def getHeaderListFromCSV(filename, delimiter):
    with io.open(filename, encoding="ISO-8859-1") as csvFile:
        try:
            buff = []
            while True:
                line = csvFile.read(1)
                buff.append(line)
                    
                if line == "" or line == '\n':
                    break
                        
            bufferString = ''.join([str(x) for x in buff])
            return bufferString.split('\t')

        except (KeyboardInterrupt, Exception) as e:
            sys.stdout.flush()
            print(e)
            print(sys.exc_info())
            pass

def scaleFeatures(listOfRows):
    return preprocessing.scale(listOfRows)

def loadCSV(filename, omitHeader=False):
    with io.open(filename, encoding="ISO-8859-1") as tsvFile:
        csvReader = csv.reader(tsvFile, delimiter=',', quotechar='\"')
        rowList = [list(row) for row in csvReader]
        
        if (omitHeader):
            rowList.pop(0)
            
        return rowList
            
def maxVector(inVector):
    maxIndex = 0
    maxValue = -999999
    for i in range(0, len(inVector)):
        if inVector[i] > maxValue:
            maxIndex = i
            maxValue = inVector[i]
            
    for i in range(0, len(inVector)):
        if i == maxIndex:
            inVector[i] = 1
        else:
            inVector[i] = 0
            
    return inVector
    
def main(argv):
    X = loadCSV(argv[1], omitHeader=True) #load the un-scaled data set with classificatioins
    Xheaders = getHeaderListFromCSV(argv[1], ",")
    #print(Xheaders)
    #print(X)
    
    #extract classifications from X
    Y = []
    for row in X:
        Y.append(row[len(row) - 1])
        
    #print(Y)
    
    #remove classifications (and class description) from X
    for i in range(0,len(X)):
        X[i] = X[i][:-2]
        
    #print(X)
    Xprepared = scaleFeatures(X)
    #print(Xprepared)
    
    #replace Y textual labels with 2-dimensional vector for classifications
    for i in range(0, len(Y)):
        if (Y[i] == "Normal"):
            Y[i] = [1,0]
        else:
            Y[i] = [0,1]
    
    #print(Y)
    Y = numpy.asarray(Y)
    
    #construct the neural network 
    numFeatures = len(Xprepared[0])
    #architectureTuple = (numFeatures, int(math.floor((3/4)*numFeatures)), int(math.floor((1/2)*numFeatures)))
    architectureTuple = (numFeatures, int(math.floor((3/4)*numFeatures)), int(math.floor((1/2)*numFeatures)), int(math.floor((1/4)*numFeatures)))
    #architectureTuple = (numFeatures * 2, numFeatures, int(math.floor((3/4)*numFeatures)), int(math.floor((1/2)*numFeatures)), int(math.floor((1/4)*numFeatures)))
    print("Hidden layer sizes: " + str(architectureTuple))
    #mlp = MLPRegressor(activation='relu', hidden_layer_sizes=architectureTuple, max_iter=1000)
    mlp = MLPClassifier(activation='relu', hidden_layer_sizes=architectureTuple, max_iter=1000)
    
    #split the training data into k-Folds for training/testing and train on each of them
    nFolds = 10
    kf = KFold(n_splits=nFolds)
    correct = 0
    outOf = 0
    for train_index, test_index in kf.split(Xprepared):
        Xtrain, Xtest = Xprepared[train_index], Xprepared[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        outOf += len(Ytest) #average accuracy will be calculated as (correct / outOf)
        mlp.fit(Xtrain, Ytrain)
        
        response = mlp.predict(Xtest)
        for i in range(0,len(Xtest)):
            #response[i] = maxVector(response[i]) #custom output layer activation function
            #print(response[i])
            if numpy.equal(response[i], Ytest[i])[0]:
                correct += 1
            
    print("Accuracy (" + str(nFolds) + "-fold):" + str(correct / outOf))
    

if __name__ == "__main__":
    main(sys.argv)