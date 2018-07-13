from __future__ import print_function

from subprocess import check_call
#import pydot
#import pydotplus
import weka.core.jvm as jvm

from weka.classifiers import *

import time

import itertools

from weka.core.converters import Loader


import StringIO

import traceback

from sys import platform

from weka.core.classes import Random

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


#from sklearn.lda import LDA
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

#from pcapng import FileScanner

##from pycopkmeans import *
##from constrained_kmeans import *
#from cop_kmeans import *


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


        



########################################## Window Extractor! ###############################################################    
## Parameters: Range starts at 0
col_time = 0
col_x = 1
col_y = 2
col_z = 3
col_user = 9
col_class = 12
win_start = 1


if platform == 'win32':
    pathSep = "\\"
    pathSepD = "\\"
else:
    pathSep = "/"
    pathSepD = "/"
    
input_file_app = "packet_user.csv"

#user_start = 100104
#user_end = 100111

## Look into variable these
window_size = 96
shift_amount = 32

## 0 if just set; 1 if whole range
window_study = 0

if (window_study == 0):
    a1 = window_size
    a2 = shift_amount
else:
    a1 = 1
    a2 = 1

if not os.path.exists("output"):
        os.makedirs("output")

#for user in range(user_start, user_end + 1):

                
twin_start = win_start
for a in range(a1,window_size+1):
    for b in range(a2,shift_amount+1):
        #output_file = "out\win_" + str(a) + "_shift_" + str(b) + "_user_" + str(user) + ".csv"       
        output_file = "output" + pathSep + "win_" + str(a) + "_shift_" + str(b) + ".csv"       
        o = open( output_file, 'wt' )
        
        for input_file in glob.glob('*.csv'): 
            if (input_file != input_file_app):
                print (input_file)
                           
                result=np.array(list(csv.reader((open(input_file, "rb")), delimiter=",")))
                ##if (str(100106) in result[1][col_user] or str(100107) in result[1][col_user] or str(100109) in result[1][col_user]):           
                
                ##print result
                twin_start = win_start              
                
                while((twin_start + a) <= len(result)):
                    o.write(result[twin_start][col_time] + ", " + result[twin_start + a - 1][col_time]  + ", ")
                    for x in range(twin_start, twin_start + a):
                            o.write(result[x][col_x] + ", ")
                    for x in range(twin_start, twin_start + a):
                            o.write(result[x][col_y] + ", ")
                    for x in range(twin_start, twin_start + a):
                            o.write(result[x][col_z] + ", ")
                    o.write(stats.mode(np.take(result[:, col_class], np.arange(twin_start, twin_start + a)))[0])
                    o.write(", ")
                    o.write(stats.mode(np.take(result[:, col_user], np.arange(twin_start, twin_start + a)))[0])
                    o.write("\n")
                    twin_start = twin_start + b 
                    
        
        o.close()
        
    
########################################## Feature Extractor! ###############################################################  

print ("Beginning Feature Extraction! This may take a few minutes!") 


#for input_file in glob.glob('*.csv'): 
input_file = output_file
result=np.array(list(csv.reader((line.replace('\0','') for line in open(input_file, "rb")), delimiter="," )))

timeArr = result[:,0:2] 

result = result[:,2:len(result[0])]

#print (result)

out, labels, users = calculateFeatures(result)

output_file = "output" + pathSepD + "features_" + "win_" + str(a) + "_shift_" + str(b) + ".csv"
input_file =  "features_" + "win_" + str(a) + "_shift_" + str(b) + ".csv"
o = open( output_file, 'wt' )
    
for a in range(len(out)):
    o.write(str(timeArr[a][0]) + ", " + str(timeArr[a][1]) + ", ")
    for b in range (len(out[a])):
        o.write(str(out[a][b]) + ", ")
    o.write(str(labels[a]) + ", " + str(users[a]) + "\n")   

o.close()
 

print ("Beginning Feature Analysis! This may take a few minutes!") 

###############################  K-means ###################

input_folder = "output" ################# INSERT FEATURE EXTRACTED CSV FILE HERE!!!!! ######
print ("************************")
print (input_file)
org = input_file.rpartition('.')[0]

X=np.array(list(csv.reader((line.replace('\0','') for line in open(os.getcwd() + pathSepD + input_folder + pathSepD + input_file, "rb")), delimiter="," )))

X1 = np.array(list([X[X[:,len(X[0])-2]==k] for k in np.unique(X[:,len(X[0])-2])]))
    
true_user = []
true_time = []

n_cluster_ = {'Flashlight' : 5, 'None' : 1, 'Shoulder Radio' : 6, 'Transition' : 5, 'Whistle' : 6, 'Examine Eyes' : 5, 'High Wave' : 5, 'Take BP' : 4}

yon = raw_input('Limit number of supports? Y or N: ')
    
if (yon.lower() == "y"):
    n_max_activities = int(raw_input('Maximum number of supports: '))
else:
    n_max_activities = -1    

X_ = {}
Y_ = {}
U_ = {}
T_ = {}

include_label_bools = {}

for a in range(len(X1)):
    label = X1[a][0][len(X1[a][0])-2].lstrip()
    
    if (n_max_activities > -1 and n_max_activities < len(X1[a])):
        X1[a] = np.delete(X1[a], s_[n_max_activities:len(X1[a])], 0)
    
    if ('None' in label):
        yon = 'y'
    else:
        yon = raw_input('Include label ' + label + '? Y or N: ')
    
    if (yon.lower() == "y"):
        include_label_bools[label] = True
    else:
        include_label_bools[label] = False
    
    X_[label] = X1[a][:,2:len(X1[a][0])]
    true_time.append(X1[a][:,0:2])
    
##==============================================================================
##         if (len(X_eyes) > n_max_activities):
##             np.random.shuffle(X_eyes)
##             X_eyes = X_eyes[0:n_max_activities]   
##==============================================================================
    
    true_user.append(X_[label][:,len(X_[label][0])-1])        
        
    T_[label] = X1[a][:,0:2]
    U_[label] = X1[a][:,len(X1[a][0])-1]
    
    X_[label] = np.delete(X_[label], len(X_[label][0])-1, 1)
    X_[label] = np.delete(X_[label], len(X_[label][0])-1, 1)
    X_[label] = X_[label].astype(np.float)
    X_[label] = preprocessing.scale(X_[label])
    Y_[label] = KMeans(n_clusters=n_cluster_[label], random_state=0).fit(X_[label]).predict(X_[label])
    
X = []
Y = []
T = []
U = []

############################ Unlabel Activities ##########################

none_offset = n_cluster_['None'] + 1

for key in list(Y_):
    Y_[key] = map(str, Y_[key])
    for a in range(len(Y_[key])):
        if (include_label_bools[key]):
            Y_[key][a] = key + " " + str(Y_[key][a])
        else:
            Y_[key][a] = "None " + str(int(Y_[key][a]) + none_offset)
    if not (include_label_bools[key]):
        none_offset = none_offset + n_cluster_[key] + 1
    X.append(X_[key])
    Y.append(Y_[key])
    T.append(T_[key])
    U.append(U_[key])
    
#==============================================================================
# # k means determine k
# distortions = []
# K = range(1,10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X[0])
#     kmeanModel.fit(X[0])
#     distortions.append(sum(np.min(cdist(X[0], kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(X[0]))
#  
# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k for ' + Y[0][0])
# plt.show()
#==============================================================================

X2 = np.array(X)

T2 = np.array(true_time)

if (len(X2) > 1):
    X = np.vstack((X2[0],X2[1]))
for a in range(2, len(X2)):
    X = np.vstack((X,X2[a]))    
    

#==============================================================================
# if (len(T2) > 1):
#     true_time = np.vstack((T2[0],T2[1]))
# for a in range(2, len(T2)):
#     true_time = np.vstack((true_time,T2[a]))
#==============================================================================
    
true_time = [item for sublist in true_time for item in sublist]
true_time = np.array(true_time)    


Y = [item for sublist in Y for item in sublist]
Y = np.array(Y)

true_user = [item for sublist in true_user for item in sublist]
true_user = np.array(true_user)

T = [item for sublist in T for item in sublist]
T = np.array(T)

U = [item for sublist in U for item in sublist]
U = np.array(U)


X = np.column_stack((T,X))
X = np.column_stack((X,Y))
X = np.column_stack((X,U))

################ Remove None ##################

#indexes = []
#
#for a in range(len(X)):
#    if ('None' in X[a][len(X[0])-2]):
#        indexes.append(a)
#    
#X = np.delete(X, indexes, axis=0)


for a in range(3):
    np.random.shuffle(X)
    #np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X)


true_labels = X[:,len(X[0])-2]
true_user = X[:,len(X[0])-1]
true_time = X[:,0:2]
X = np.delete(X, len(X[0])-1, 1)
X = np.delete(X, len(X[0])-1, 1)
X = np.delete(X, 0, 1)
X = np.delete(X, 0, 1)
X = X.astype(np.float)
#X = preprocessing.scale(X)

    
# Cross- Validation Schemes
kf = KFold(len(true_labels), 5, shuffle=False)
loso = LeaveOneLabelOut(true_user)
loao = LeaveOneLabelOut(true_labels)

   
# Classifier Selection
#clf = LDA()
#clf = LogisticRegression()
#clf = KNeighborsClassifier(n_neighbors=5)
#clf = svm.SVC(kernel="linear")
#clf = svm.SVC(kernel="rbf")
#clf = tree.DecisionTreeClassifier()
#clf = GaussianNB()
clf = RandomForestClassifier(oob_score=True)


#clf = AdaBoostClassifier(n_estimators=1000)
#clf = AdaBoostClassifier(tree.DecisionTreeClassifier())



# Needed for RBF SVM
#features = preprocessing.scale(features)

#cluster_score = []
#
#for a in range (2,10):
#    true_labels = KMeans(n_clusters=a, random_state=0).fit(X).predict(X)
#    allPredictions = []
#    allClassifications = []
#    for train, test in kf:        
#        clf.fit(X[train], true_labels[train])
#        predictions = clf.predict(X[test])
#        allPredictions.append( predictions)
#        allClassifications.append(true_labels[test])
#    
#    allPredictions = np.concatenate(allPredictions)
#    allClassifications = np.concatenate(allClassifications)  
#    cluster_score.append(accuracy_score(allClassifications,allPredictions))
#
#print (cluster_score)


#true_labels = KMeans(n_clusters=5, random_state=0).fit(X).predict(X)


allPredictions = []
allClassifications = []

#print(cluster_labels)
#cluster_labels = np.array(cluster_labels)
#print(cluster_labels)

for train, test in kf:
    clf.fit(X[train], true_labels[train])
    predictions = clf.predict(X[test])
    allPredictions.append( predictions)
    allClassifications.append(true_labels[test])

allPredictions = np.concatenate(allPredictions)
allClassifications = np.concatenate(allClassifications) 

for a in range(len(allPredictions)):
    for key in list(include_label_bools):
        if (key in allPredictions[a]):
            allPredictions[a] = key
            break
    
for a in range(len(allClassifications)):
    for key in list(include_label_bools):
        if (key in allClassifications[a]):
            allClassifications[a] = key
            break
   
print ('Get Results')

print(classification_report(allClassifications,allPredictions))

print("Accuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))

cnf_matrix = confusion_matrix(allClassifications,allPredictions)

np.set_printoptions(threshold='nan')

plot_confusion_matrix(cnf_matrix, classes=[], normalize=True, title='Normalized confusion matrix')