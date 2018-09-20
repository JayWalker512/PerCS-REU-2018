#!/usr/bin/env python3

from __future__ import print_function

from subprocess import check_call

import time

import itertools

import StringIO

import traceback

from sys import platform

import os
import csv
import numpy as np

from scipy import stats

import glob
from scipy import fftpack

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.semi_supervised import label_propagation

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import LeaveOneLabelOut

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from scipy.spatial.distance import cdist

from datetime import datetime

import operator

import random

def calculateFeatures(data): # put this parameter in classification
    
#    features = np.zeros((2722,33))
#
#    for i in range(0,2722):
#        
#        xData = (data[i][0:128])
#        yData = (data[i][128:256])
#        zData = (data[i][256:384])
        
    labels = data[:,len(data[0])-2]
    users = data[:,len(data[0])-1]
    data = np.delete(data, len(data[0])-1, 1)
    data = np.delete(data, len(data[0])-1, 1)
    #print ("Here") 

    data = np.asfarray(data, dtype='float')    
    
    
    nData = len(data)
    nwindow = len(data[0])    
    
    features = np.zeros((nData,72))
 
 
    for i in range(0,nData):
         
        xData = (data[i][0:int(nwindow/3)])
        yData = (data[i][int(nwindow/3):int((nwindow/3)*2)])
        zData = (data[i][int((nwindow/3)*2):nwindow])
         
        ##########################################
        # Feature Subect 1: Time Domain Features #        
        ##########################################
         
        # Calculate the Min
        minX = np.min(xData)
        minY = np.min(yData)
        minZ = np.min(zData)
        features[i][0:3] = [minX, minY, minZ]



     
        # Max
        maxX = np.max(xData)
        maxY = np.max(yData)
        maxZ = np.max(zData)
        features[i][3:6] = [maxX, maxY, maxZ]
        #print features[nData-1][3:6]



     
        # Mean
        meanX = np.mean(xData)
        meanY = np.mean(yData)
        meanZ = np.mean(zData)
        features[i][6:9] = [meanX, meanY, meanZ]



         
        # Standard Deviation
        stdDevX = np.std(xData)
        stdDevY = np.std(yData)
        stdDevZ = np.std(zData)
        features[i][9:12] = [stdDevX, stdDevY, stdDevZ]  
        #print features[nData-1][9:12]

 
         
        # Pairwise Correlation
        corrXY = np.correlate(xData,yData)
        corrXZ = np.correlate(xData,zData)
        corrYZ = np.correlate(yData,zData)
        features[i][12:15] = [corrXY, corrXZ, corrYZ]  
        #print features[nData-1][12:15]
 
 
        # Zero Crossing-Rate
        zeroRateX = zero_cross_rate(xData)
        zeroRateY = zero_cross_rate(yData)
        zeroRateZ = zero_cross_rate(zData)
        features[i][15:18] = [zeroRateX, zeroRateY, zeroRateZ]


        #Skew
        xskew = stats.skew(xData)
        yskew = stats.skew(yData)
        zskew = stats.skew(zData)
        features[i][18:21] = [xskew, yskew, zskew]


         
        #Kurtosis
        xkurt = stats.kurtosis(xData)
        ykurt = stats.kurtosis(yData)
        zkurt = stats.kurtosis(zData)
        features[i][21:24] = [xkurt, ykurt, zkurt]


        #Signal-to-noise ratio
        xsnr = features[i][6]/features[i][9]
        ysnr = features[i][7]/features[i][10]
        zsnr = features[i][8]/features[i][11]
        features[i][24:27] = [xsnr, ysnr, zsnr]
        
         
        #Mean cross rate
        xMCR = mean_cross_rate(xData)
        yMCR = mean_cross_rate(yData)
        zMCR = mean_cross_rate(zData)
        features[i][27:30] = [xMCR, yMCR, zMCR]
        
         
        #Trapezoidal area
        xArea = np.trapz(xData)
        yArea = np.trapz(yData)
        zArea = np.trapz(zData)
        features[i][30:33] = [xArea, yArea, zArea]
        
        
        xFFT = np.fft.fft((xData))
        yFFT = np.fft.fft((yData))
        zFFT = np.fft.fft((zData))
        
        xSigEnergy = (1.0/128.0)*sum(np.power(np.absolute(xFFT),2))
        ySigEnergy = (1.0/128.0)*sum(np.power(np.absolute(yFFT),2))
        zSigEnergy = (1.0/128.0)*sum(np.power(np.absolute(zFFT),2))
        features[i][33:36] = [xSigEnergy, ySigEnergy, zSigEnergy]
        
        
        xFFTFreq = fftpack.fftfreq(xFFT.size, 1.0/50.0)
        yFFTFreq = fftpack.fftfreq(yFFT.size, 1.0/50.0)
        zFFTFreq = fftpack.fftfreq(zFFT.size, 1.0/50.0)
        #print features[nData-1][36:40] 
        
        xTopFreqs = getTopFreqs(xFFT, xFFTFreq)
        yTopFreqs = getTopFreqs(yFFT, yFFTFreq)
        zTopFreqs = getTopFreqs(zFFT, zFFTFreq)
        features[i][36:48] = np.concatenate([xTopFreqs,yTopFreqs,zTopFreqs])
    
        
        # FFT
        # Sampling Rate 50Hz
        # Nyquist Frequency 25Hz        
        features[i][48:72] = np.concatenate([np.absolute(xFFT[0:8]),np.absolute(yFFT[0:8]),np.absolute(zFFT[0:8])])

     


    return features, labels, users



def zero_cross_rate(X):
    count = 1
    cross = 0
    while count < len(X):
        if X[count-1] > 0 and X[count] < 0:
            cross+=1
            count+=1
        elif X[count-1] < 0 and X[count]> 0:
            cross+=1
            count+=1
        else:
            count+=1
         
    return cross
 
 
 
def mean_cross_rate(X):
  mean = np.mean(X)
  count = 1
  cross = 0
  while count < len(X):
        if X[count-1] > mean and X[count] < mean:
            cross+=1
            count+=1
        elif X[count-1] < mean and X[count]> mean:
            cross+=1
            count+=1
        else:
            count+=1
         
  return cross



def getTopFreqs(fftData,fftFreq):
    ind = np.argpartition(np.abs(fftData)**2, -4)[-4:]
    freqs = fftFreq[ind[np.argsort(fftData[ind])]]
        
    return freqs

##########################################################################################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    print ("\n")