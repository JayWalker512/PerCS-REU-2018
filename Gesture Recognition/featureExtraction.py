# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 18:56:17 2014

@author: Johnny
"""

import numpy as np;

import scipy.fftpack as fftpack



import scipy.stats as stats

#import pywt




def main(data,classifications):



    features = np.zeros((len(data),60))
    nwindow = len(data[0])  

    for i in range(0,len(data)):
        
        xData = (data[i][0:int(nwindow/3)])
        yData = (data[i][int(nwindow/3):int((nwindow/3)*2)])
        zData = (data[i][int((nwindow/3)*2):nwindow])
        
      
        
        ##########################################
        # Feature Subect 1: Time Domain Features #        
        ##########################################
        
        # Min        
        minX = np.min(xData)
        minY = np.min(yData)
        minZ = np.min(zData)
        features[i][0:3] = [minX, minY, minZ]
    
        # Max
        maxX = np.max(xData)
        maxY = np.max(yData)
        maxZ = np.max(zData)    
        features[i][3:6] = [maxX, maxY, maxZ]
    
        # Mean
        meanX =  np.mean(xData)
        meanY =  np.mean(yData)
        meanZ =  np.mean(zData)
        features[i][6:9] = [meanX, meanY, meanZ]        
        
        # Standard Deviation
        stdDevX =  np.std(xData)
        stdDevY =  np.std(yData)
        stdDevZ =  np.std(zData)
        features[i][9:12] = [stdDevX, stdDevY, stdDevZ]  
        
        # Pairwise Correlation
        
        corrXY = np.corrcoef(xData,yData)[0][1]
        corrXZ = np.corrcoef(xData,zData)[0][1]
        corrYZ = np.corrcoef(yData,zData)[0][1]
        features[i][12:15] = [corrXY, corrXZ, corrYZ]  

        # Zero Crossing-Rate
        zeroRateX = zero_cross_rate(xData)
        zeroRateY = zero_cross_rate(yData)
        zeroRateZ = zero_cross_rate(zData)
        features[i][15:18] = [zeroRateX, zeroRateY, zeroRateZ]

        xskew = stats.skew(xData)
        yskew = stats.skew(yData)
        zskew = stats.skew(zData)
        features[i][18:21] = [xskew, yskew, zskew]
        
        xkurt = stats.kurtosis(xData)
        ykurt = stats.kurtosis(yData)
        zkurt = stats.kurtosis(zData)
        features[i][21:24] = [xkurt, ykurt, zkurt]
        
        
        #Signal-to-noise ratio
        xsnr = features[i][6]/features[i][9]
        ysnr = features[i][7]/features[i][10]
        zsnr = features[i][8]/features[i][11]
        features[i][24:27] = [xsnr, ysnr, zsnr]
        
        
        xMCR = mean_cross_rate(xData)
        yMCR = mean_cross_rate(yData)
        zMCR = mean_cross_rate(zData)
        features[i][27:30] = [xMCR, yMCR, zMCR]
        
                
        xArea = np.trapz(xData)
        yArea = np.trapz(yData)
        zArea = np.trapz(zData)
        features[i][30:33] = [xArea, yArea, zArea]       
        
        
        xFFT = np.fft.fft(xData)
        yFFT = np.fft.fft(yData)
        zFFT = np.fft.fft(zData)   
        
        xSigEnergy = (1.0/128.0)*sum(np.power(np.absolute(xFFT),2))
        ySigEnergy = (1.0/128.0)*sum(np.power(np.absolute(yFFT),2))
        zSigEnergy = (1.0/128.0)*sum(np.power(np.absolute(zFFT),2))
        features[i][33:36] = [xSigEnergy, ySigEnergy, zSigEnergy]
        
        
        
        xFFTFreq = fftpack.fftfreq(xData.size, 1.0/50.0)
        
        yFFTFreq = fftpack.fftfreq(yData.size, 1.0/50.0)
        zFFTFreq = fftpack.fftfreq(zData.size, 1.0/50.0)
        
        xTopFreqs = getTopFreqs(xFFT, xFFTFreq)
        yTopFreqs = getTopFreqs(yFFT, yFFTFreq)
        zTopFreqs = getTopFreqs(zFFT, zFFTFreq)
        
        
        
        #features[i][36:48] = np.concatenate([xTopFreqs,yTopFreqs,zTopFreqs])        
        
        
        # FFT
        # Sampling Rate 50Hz
        # Nyquist Frequency 25Hz        
        
        features[i][36:60] = np.concatenate([np.absolute(xFFT[0:8]),np.absolute(yFFT[0:8]),np.absolute(zFFT[0:8])])        
  
        
    return features












def zero_cross_rate(X):
    
    counter = 0.0;    

    
    for i in range(1,len(X)):

        if ((X[i-1]*X[i]) < 0):
            counter = counter + 1
    
    
    return counter*(1.0/127.0)



def mean_cross_rate(X):
    
    counter = 0.0;   
    mean = np.mean(X)
    shiftedData = X - mean
    

    
    for i in range(1,len(X)):

        if ((shiftedData[i-1]*shiftedData[i]) < 0):
            counter = counter + 1
    
    
    return counter*(1.0/127.0)

    
    
    
def getTopFreqs(fftData,fftFreq):
    
    
        ind = np.argpartition(np.abs(fftData)**2, -4)[-4:]    
    
    
        freqs = fftFreq[ind[np.argsort(fftData[ind])]]    
        
        return freqs
        
     
def waveletFeatures(X):
    
    waveDecomps = pywt.wavedec(X, 'db8', level=3)
    
    
    allFeats = []    
    
    for decompLvl in waveDecomps:
        
        minVal = np.min(decompLvl)
        maxVal = np.max(decompLvl)
        stdDev = np.std(decompLvl)
        meanVal = np.mean(decompLvl)
        
        currFeats = [minVal, maxVal, stdDev, meanVal]
        
        allFeats = np.concatenate([allFeats, currFeats])
        
        
        
    return allFeats
        
        

    
    
    
        
        
        
        