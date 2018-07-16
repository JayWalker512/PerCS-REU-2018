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

#n_cluster_ = {'Flashlight' : 5, 'None' : 1, 'Shoulder Radio' : 6, 'Transition' : 5, 'Whistle' : 6, 'Examine Eyes' : 5, 'High Wave' : 5, 'Take BP' : 4}
n_cluster_ = {}

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
    
    #Determine Number of Clusters Here
    opt_n = 1
    max_val = -1
    for n in range (2,10):
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n)
        cluster_labels = clusterer.fit_predict(X_[label])
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_[label], cluster_labels)
        #print("For n_clusters = ", n, ", The average silhouette_score is :", silhouette_avg)
        
        if (max(max_val, silhouette_avg) > max_val):
            opt_n = n
            max_val = silhouette_avg
    
    n_cluster_[label] = opt_n
    Y_[label] = KMeans(n_clusters=n_cluster_[label]).fit(X_[label]).predict(X_[label])
    
X = []
Y = []
T = []
U = []

############################ Unlabel Activities ##########################

if 'None' in n_cluster_.keys():
    none_offset = n_cluster_['None'] + 1
else:
    none_offset =  1

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

######################################## To Flat File #####################################################################

output_file = "output" + pathSepD + "flat_" + str(window_size) + "_shift_" + str(shift_amount) + ".csv"       
o = open( output_file, 'wt' )

results = np.column_stack((true_time,true_user))
results = np.column_stack((results,allPredictions))
results = np.column_stack((results,allClassifications))

results = sorted(results, key=operator.itemgetter(2, 0))

allClassifications = []
allPredictions = []

for a in range (len(results)):
    d1 = datetime.strptime(results[a][0].strip().split("-",1)[0] + "." + results[a][0].strip().split("-",1)[1].zfill(3), "%H:%M:%S.%f")
    o.write(str(3600000 * d1.hour + 60000 * d1.minute + 1000 * d1.second + d1.microsecond/1000) + ", ")
    results[a][0] = str(3600000 * d1.hour + 60000 * d1.minute + 1000 * d1.second + d1.microsecond/1000)
    d1 = datetime.strptime(results[a][1].strip().split("-",1)[0] + "." + results[a][1].strip().split("-",1)[1].zfill(3), "%H:%M:%S.%f")
    o.write(str(3600000 * d1.hour + 60000 * d1.minute + 1000 * d1.second + d1.microsecond/1000) + ", ")
    results[a][1] = str(3600000 * d1.hour + 60000 * d1.minute + 1000 * d1.second + d1.microsecond/1000)
    o.write(str(results[a][2]))                
    o.write(", ")
    o.write(str(results[a][3]))                
    o.write("\n")
    
    allPredictions.append(str(results[a][3]))
    allClassifications.append(str(results[a][4]))
o.close()


########################################## To Rules ###############################################################    
## Parameters: Range starts at 0
col_time_start = 0
col_time_end = 1
col_user = 2
col_class = 3

col_time_app = 1
col_usage = 6
col_user_app = 7

input_folder = "output"
input_file = "flat_" + str(window_size) + "_shift_" + str(shift_amount) + ".csv"


shift_amount_seconds = int((shift_amount / 64.0) * 1000)



org = input_file.rpartition('.')[0]

output_file = "output" + pathSepD + "pre_" + input_file


X=np.array(list(csv.reader((line.replace('\0','') for line in open(os.getcwd() + pathSepD + input_folder + pathSepD + input_file, "rb")), delimiter="," )))


Y=np.array(list(csv.reader((line.replace('\0','') for line in open(input_file_app, "rb")), delimiter="," )))
Y = np.delete(Y, (0), axis=0)


X = X[np.lexsort((X[:,col_time_start],X[:,col_user]))]    
Y = Y[np.lexsort((Y[:,col_time_app],Y[:,col_user_app]))]    


########################################## Fix Transitions With Thresh ########################################################

######### Last Transition Thresh #####

#==============================================================================
# ttime = 0
# thresh = 5
# 
# for a in range (1, len(X)-1):
#     if ('Transition' in str(X[a][col_class])):
#         tempt = (int(X[a][col_time_start]) - ttime) / 1000.0 
#         if (ttime == 0): ####### First Transition Sequence ######
#             ttime == int(X[a][col_time_end])
#         elif(tempt > 0 and tempt < thresh): ##### Below Thresh #########
#             allPredictions[a] = X[a-1][col_class].strip()
#             if ('Transition' in allPredictions[a] and 'Transition' not in allPredictions[a+1]):
#                 ttime == int(X[a][col_time_end])
#         else:
#             ttime == int(X[a][col_time_end])
#==============================================================================
            
######### Transition Duration Thresh #####    

#==============================================================================
# block = False
# thresh = 2.0
# 
# for a in range (1, len(X)-1):
#     if ('Transition' in str(X[a][col_class]) and block == False):
#         block = True
#         tStart = (int(X[a][col_time_start])) / 1000.0
#         aStart = a
#     elif ('Transition' in str(X[a][col_class]) and 'Transition' not in str(X[a+1][col_class])):
#         block = False
#         aEnd = a + 1
#         tEnd = (int(X[a][col_time_start])) / 1000.0
#         if (tEnd - tStart < thresh):
#             for b in range(aStart, aEnd):
#                 allPredictions[b] = 'None'.strip()
#==============================================================================

####################################### Isolated Transition False Positive ########################################################

for a in range (1, len(X)-1):
    if (str(X[a][col_class]) not in str(X[a-1][col_class]) and str(X[a][col_class]) not in str(X[a+1][col_class]) and 'Transition'.strip() in str(X[a][col_class])):
        if (X[a-1][col_class].strip() in X[a+1][col_class].strip()):
            allPredictions[a] = X[a-1][col_class].strip()
            X[a][col_class] = X[a-1][col_class].strip()
        else:
            allPredictions[a] = 'None'.strip()
            X[a][col_class] = 'None'.strip()

########################################## Broken Block ########################################################

for a in range (1, len(X)-1):
    if (X[a+1][col_class].strip() in str(X[a-1][col_class]) and X[a-1][col_class].strip() not in str(X[a][col_class])):
        allPredictions[a] = X[a-1][col_class].strip()
        X[a][col_class] = X[a-1][col_class].strip()
        

####################################### Isolated False Positive ########################################################

for a in range (1, len(X)-1):
    if (str(X[a][col_class]) not in str(X[a-1][col_class]) and str(X[a][col_class]) not in str(X[a+1][col_class]) and 'None'.strip() not in str(X[a][col_class])):
        if (X[a-1][col_class].strip() in X[a+1][col_class].strip()):
            allPredictions[a] = X[a-1][col_class].strip()
            X[a][col_class] = X[a-1][col_class].strip()
        else:
            allPredictions[a] = 'None'.strip()
            X[a][col_class] = 'None'.strip()


########## Reanalyze Results ############

#print ('Get Results')
#
#print(classification_report(allClassifications,allPredictions))
#
#print("Accuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
#
#
#cnf_matrix = confusion_matrix(allClassifications,allPredictions)
#
#np.set_printoptions(threshold='nan')
#
#
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=[], normalize=True,                      title='Normalized confusion matrix')
#
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=[], normalize=False,                      title='Confusion matrix')


######################################### Flat File Mod ###################################

output_file2 = "output" + pathSepD + "flat_" + str(window_size) + "_shift_" + str(shift_amount) + "_mod.csv"       
o = open( output_file2, 'wt' )



for a in range (len(X)):
    o.write(X[a][0] + ", " + X[a][1] + ", " + X[a][2] + ", " + X[a][3])                  
    o.write("\n")
o.close()

####################### Inject App Usage ##########################################
app_usages = []

for a in range(len(Y)):
    d1 = datetime.strptime(Y[a][col_time_app].split(".",1)[0] + "." + Y[a][col_time_app].split(".",1)[1].zfill(3), "%H:%M:%S.%f")
    Y[a][col_time_app] = (str(3600000 * d1.hour + 60000 * d1.minute + 1000 * d1.second + d1.microsecond/1000 - 2500))
    break_X = False
    for b in range(len(X)):
        if (break_X):
            pass
        elif (Y[a][col_user_app] in X[b][col_user] and int(X[b][col_time_start]) <= int(Y[a][col_time_app]) <= int(X[b][col_time_end])):
            
            X[b][col_time_end] = int(Y[a][col_time_app]) - 1
            if (b != len(X) - 1):
                X[b+1][col_time_start] = int(Y[a][col_time_app]) + 1
            X = np.insert(X, b+1, np.array((Y[a][col_time_app], Y[a][col_time_app], Y[a][col_user_app], Y[a][col_usage])), 0)  
            allClassifications = np.insert(allClassifications, b+1,Y[a][col_usage], 0)
            
            break_X = True
            
            if (Y[a][col_usage].lstrip() not in (app_usages)):
                app_usages.append(Y[a][col_usage])
            
        

####################### Create Activity Blocks ##########################################


a = 0
thresh = 10 * 1000
#print (thresh)
len_x = len(X)

o = open( output_file, 'wt' )

while (a < len_x):
    temp_a = a + 1
    diff = False
    while (temp_a < len_x and diff == False):
        #abs(int(X[temp_a][0]) - int(X[temp_a - 1][1])) > thresh or 
        #X[temp_a][2] != X[a][2]
        # or X[temp_a][3] != X[temp_a - 1][3]
        if (abs(int(X[temp_a][col_time_start]) - int(X[temp_a - 1][col_time_end])) > thresh or X[temp_a][col_user].strip() not in X[a][col_user].strip() or X[temp_a][col_class].strip() not in X[temp_a - 1][col_class].strip()):            
            o.write(X[a][col_time_start] + ", " + X[temp_a - 1][col_time_end] + ", " + X[a][col_user] + ", " + X[a][col_class] + "\n")
            a = temp_a
            diff = True
        else:
            temp_a = temp_a + 1
    
    if (diff == False):
        o.write(X[a][col_time_start] + ", " + X[temp_a - 1][col_time_end] + ", " + X[a][col_user] + ", " + X[a][col_class] + "\n")
        a = len_x
o.close()


Z=np.array(list(csv.reader((line.replace('\0','') for line in open(os.getcwd() + pathSepD + input_folder + pathSepD + "pre_" + input_file, "rb")), delimiter="," )))


########################## CALCULATE DURATION RANGES ##################

min_time = int(Z[0][col_time_start])
max_time = int(Z[len(Z)-1][col_time_end])

n_ = {}
delta_t_ = {}

for a in range(len(Z)):
    label = Z[a][col_class].lstrip()
    
    if label in n_.keys():
        n_[label] = n_[label] + 1
        delta_t_[label] = delta_t_[label] + (int(Z[a][col_time_end]) - int(Z[a][col_time_start]))
    else:
        n_[label] =  1
        delta_t_[label] = int(Z[a][col_time_end]) - int(Z[a][col_time_start])

for key in list(n_):
    delta_t_[key] = delta_t_[key] / n_[key]
    


########################## START INJECT SIMULATED USERS HERE ##################

#    ########################## User 100110 - Walking Data #####################
#delta_t = random.randint(min_time, max_time)
#delta_t2 = delta_t
#uid = "100110"
#add = "Walking "
#
#
#Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#delta_t = int(Z[len(Z) - 1][col_time_end])
#
#for a in range(25):        
#    ######################## SEQUENCE ONE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), uid, add + "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), uid, "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#       
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), uid, add + "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), uid, add + "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), uid, add + "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    ########################## User 100111 - Walking Data #####################
#delta_t = random.randint(min_time, max_time)
#uid = "100111"
#add = "Walking "
#
#
#Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#delta_t = int(Z[len(Z) - 1][col_time_end])
#
#for a in range(25):        
#    ######################## SEQUENCE ONE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), uid, add + "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), uid, "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#       
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), uid, add + "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), uid, add + "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), uid, add + "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#        
#
#
#
#
########################### User 200101 - App Usage Attack #####################
#delta_t = random.randint(min_time, max_time)
#
#Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#delta_t = int(Z[len(Z) - 1][col_time_end])
#
#
#for a in range(5):
#    ######################## SEQUENCE ONE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), "200101", "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), "200101", "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Requesting Backup"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), "200101", "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), "200101", "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    ######################## SEQUENCE TWO ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), "200101", "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), "200101", "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Vitals Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), "200101", "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), "200101", "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    ######################## SEQUENCE THREE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), "200101", "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Vitals Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), "200101", "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), "200101", "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Requesting Backup"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), "200101", "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    ######################## SEQUENCE FOUR ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), "200101", "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Requesting Backup"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), "200101", "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_high_wave - 2000, delta_t_high_wave + 2000)), "200101", "High Wave"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_examine_eyes - 2000, delta_t_examine_eyes + 2000)), "200101", "Examine Eyes"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_take_BP - 2000, delta_t_take_BP + 2000)), "200101", "Take BP"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Vitals Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])  
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), "200101", "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    ######################## SEQUENCE FIVE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), "200101", "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), "200101", "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#       
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), "200101", "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), "200101", "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), "200101", "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), "200101", "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    
#        ########################## User 200102 - Walking Variance GPS Attack #####################
#delta_t = random.randint(min_time, max_time)
#uid = "200102"
#add = "Walking "
#
#
#Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#delta_t = int(Z[len(Z) - 1][col_time_end])
#
#for a in range(15):        
#    ######################## SEQUENCE ONE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), uid, add + "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), uid, "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#       
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), uid, add + "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), uid, add + "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), uid, add + "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#        
#
#    ########################## User 200103 - Stationary Drift GPS Attack #####################
#delta_t = random.randint(min_time, max_time)
#uid = "200103"
#add = ""
#
#
#Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#delta_t = int(Z[len(Z) - 1][col_time_end])
#
#for a in range(15):        
#    ######################## SEQUENCE ONE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), uid, add + "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), uid, "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#       
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), uid, add + "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), uid, add + "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), uid, add + "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#
#
#    ########################## User 100110 - Walking Data #####################
#delta_t = random.randint(min_time, max_time)
#delta_t2 = delta_t
#uid = "500101"
#add = " "
#
#
#Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#delta_t = int(Z[len(Z) - 1][col_time_end])
#
#for a in range(25):        
#    ######################## SEQUENCE ONE ######################
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_shoulder_radio - 2000, delta_t_shoulder_radio + 2000)), uid, add + "Shoulder Radio"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t + 1
#    Z = np.vstack((Z, [str(delta_t), str(delta_t), uid, "Location Sent"]))
#    delta_t = delta_t + 1
#    
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#       
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_flashlight - 2000, delta_t_flashlight + 2000)), uid, add + "Flashlight"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_whistle - 2000, delta_t_whistle + 2000)), uid, add + "Whistle"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])    
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_transition - 2000, delta_t_transition + 2000)), uid, add + "Transition"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])
#    
#    delta_t = delta_t - shift_amount_seconds
#    Z = np.vstack((Z, [str(delta_t), str(delta_t + random.randint(delta_t_none - 2000, delta_t_none + 2000)), uid, add + "None"]))
#    delta_t = int(Z[len(Z) - 1][col_time_end])


########################## END INJECT SIMULATED USERS HERE ####################
########################## FIND SEQUENCES ####################

a = 0
len_z = len(Z)

output_file = "output" + pathSepD + "sequence_" + input_file

arr = []

column_names = []


while (a < len_z):
    temp_a = a + 2
    diff = False
    breaku = False
    while (temp_a < len_z and diff == False and breaku == False):
        if (Z[temp_a][col_user].strip() != Z[a][col_user].strip()):
            diff = True
        elif ("Transition" in Z[temp_a][col_class] and temp_a > 1):   #Hardcoded Sequence Split     
            arr.append([])
            arr[len(arr)-1].append({})
            arr[len(arr)-1].append({"Distance Magnitude": 0, "Walking":"0","App Usage Count": 0, "User":Z[a][col_user], "Anomaly":"Normal"})
                
            for b in range(a,temp_a):
                label = Z[b][col_class].lstrip()
                if ("None" not in label and "Transition" not in label):                    
                    if ("Walking" in label):
                        arr[len(arr)-1][1]["Walking"] = "1"
                        label = label.replace("Walking ","")
                                        
                    arr[len(arr)-1][0][label] = "1"                    
                    
                    if (label not in app_usages):                    
                        start = label + " Start"
                        end = label + " End"
                        delta = label + " Delta"
                        
                        arr[len(arr)-1][0][start] = Z[b][col_time_start]
                        arr[len(arr)-1][0][end] = Z[b][col_time_end]
                        arr[len(arr)-1][0][delta] = str((float(Z[b][col_time_end]) - float(Z[b][col_time_start]))/1000)   
                        
                        if (label not in column_names):
                            column_names.append(label)
                    else:
                        arr[len(arr)-1][1]["App Usage Count"] = arr[len(arr)-1][1]["App Usage Count"] + 1
                    
                    
                    
            #if(not("1" in arr[len(arr)-1][0] and "1" in arr[len(arr)-1][1] and "1" in arr[len(arr)-1][2] and "0" in arr[len(arr)-1][3] and "0" in arr[len(arr)-1][4] and "0" in arr[len(arr)-1][5] and "1" in arr[len(arr)-1][6] and "0" in arr[len(arr)-1][7] and "0" in arr[len(arr)-1][8]) and not("1" in arr[len(arr)-1][0] and "1" in arr[len(arr)-1][1] and "1" in arr[len(arr)-1][2] and "1" in arr[len(arr)-1][3] and "1" in arr[len(arr)-1][4] and "1" in arr[len(arr)-1][5] and "1" in arr[len(arr)-1][6] and "1" in arr[len(arr)-1][7] and "1" in arr[len(arr)-1][8])):
            #if(("1" in arr[len(arr)-1][8] or "1" in arr[len(arr)-1][9]) and ("0" in arr[len(arr)-1][3] and "0" in arr[len(arr)-1][4] and "0" in arr[len(arr)-1][5])):
#                arr[len(arr)-1][len(arr[0])-1] = "Abnormal"
            
            if (len(list(arr[len(arr)-1][0])) == 0):
                arr.pop(len(arr)-1)
            
            a = temp_a
            breaku = True
        else:
            temp_a = temp_a + 1
    
    if (breaku == False):
        #o.write(X[a][col_time_start] + ", " + X[temp_a - 1][col_time_end] + ", " + X[a][col_user] + ", " + X[a][col_class] + "\n")
        arr.append([])
        arr[len(arr)-1].append({})
        arr[len(arr)-1].append({"Distance Magnitude": "0", "Walking":"0","App Usage Count": 0, "User":Z[a][col_user], "Anomaly":"Normal"})
        
        for b in range(a,temp_a):
            label = Z[b][col_class].lstrip()
            if ("None" not in label and "Transition" not in label):                    
                if ("Walking" in label):
                    arr[len(arr)-1][1]["Walking"] = "1"
                    label = label.replace("Walking ","")
                
                arr[len(arr)-1][0][label] = "1"                    
                    
                if (label not in app_usages):                    
                    start = label + " Start"
                    end = label + " End"
                    delta = label + " Delta"
                    
                    arr[len(arr)-1][0][start] = Z[b][col_time_start]
                    arr[len(arr)-1][0][end] = Z[b][col_time_end]
                    arr[len(arr)-1][0][delta] = str((float(Z[b][col_time_end]) - float(Z[b][col_time_start]))/1000)   
                    
                    if (label not in column_names):
                        column_names.append(label)
                else:
                    arr[len(arr)-1][1]["App Usage Count"] = arr[len(arr)-1][1]["App Usage Count"] + 1
                
                
                                
        #o.write(X[a][col_time_start] + ", " + X[temp_a - 1][col_time_end] + ", " + X[a][col_user] + ", " + X[a][col_class] + "\n")
        
        #if(not("1" in arr[len(arr)-1][0] and "1" in arr[len(arr)-1][1] and "1" in arr[len(arr)-1][2] and "0" in arr[len(arr)-1][3] and "0" in arr[len(arr)-1][4] and "0" in arr[len(arr)-1][5] and "1" in arr[len(arr)-1][6] and "0" in arr[len(arr)-1][7] and "0" in arr[len(arr)-1][8]) and not("1" in arr[len(arr)-1][0] and "1" in arr[len(arr)-1][1] and "1" in arr[len(arr)-1][2] and "1" in arr[len(arr)-1][3] and "1" in arr[len(arr)-1][4] and "1" in arr[len(arr)-1][5] and "1" in arr[len(arr)-1][6] and "1" in arr[len(arr)-1][7] and "1" in arr[len(arr)-1][8])):
        #if(("1" in arr[len(arr)-1][8] or "1" in arr[len(arr)-1][9]) and ("0" in arr[len(arr)-1][3] and "0" in arr[len(arr)-1][4] and "0" in arr[len(arr)-1][5])):
#            arr[len(arr)-1][len(arr[0])-1] = "Abnormal"
        if (len(list(arr[len(arr)-1][0])) == 0):
            arr.pop(len(arr)-1)
            
        a = temp_a

######################### Calculate Distance Magnitude #######################################
for a in range(len(arr)):
    if ("0" in arr[a][1]["Walking"]): #if walking
        arr[a][1]["Distance Magnitude"] = str(random.uniform(0.0, 3.0))
    else:
        seq_start = 4687389847 # Some Date in 2118
        seq_end = -1
        for keys in list(arr[a][0]):
            if ("Start" in keys):
                seq_start = min(seq_start, float(arr[a][0][keys]))
            if ("End" in keys):
                seq_end = max(seq_end, float(arr[a][0][keys]))
        if (seq_end < 0):
            seq_start = 0 # Some Date in 2118
            seq_end = 0
        arr[a][1]["Distance Magnitude"] = str((float(seq_end) - float(seq_start)) * (1.39/1000.0))
        
########################## Inject GPS Attacks #######################################
#for a in range(len(arr)):
#    
#    #################### Walking Variance GPS Attack ############################
#    if ("200102" in arr[a][len(arr[0])-2]):
#        arr[a][28] = str((float(arr[a][17]) - float(arr[a][10])) * (2.78/1000.0))
#        arr[a][len(arr[0])-1] = "Abnormal"
##################### Stationary Drift GPS Attack ############################
#    elif ("200103" in arr[a][len(arr[0])-2]):
#        arr[a][28] = str((float(arr[a][17]) - float(arr[a][10])) * (.695/1000.0))
#        arr[a][len(arr[0])-1] = "Abnormal"   
#        
##################### DOS Attack ############################
#    elif ("500101" in arr[a][len(arr[0])-2]):
#        arr[a][29] = str(random.randint(5, 10))
#        arr[a][len(arr[0])-1] = "Abnormal"   

################ Convert Dictionaries to Array #######

seqs = []
seqs.append([])

for names in column_names:
    seqs[0].append(names)
    seqs[0].append(names + " Start")
    seqs[0].append(names + " End")
    seqs[0].append(names + " Delta")

for names in app_usages:
    seqs[0].append(names)

seqs[0].append("Distance Magnitude")
seqs[0].append("Walking")
seqs[0].append("App Usage Count")
seqs[0].append("User")
seqs[0].append("Anomaly")

for a in range (len(arr)):
    seqs.append([])
    for b in range (len(seqs[0]) - 5):
        if (seqs[0][b] in arr[a][0]):
            seqs[len(seqs)-1].append(arr[a][0][seqs[0][b]])
        else:
            seqs[len(seqs)-1].append("0")
    seqs[len(seqs)-1].append(arr[a][1]["Distance Magnitude"])
    seqs[len(seqs)-1].append(arr[a][1]["Walking"])
    seqs[len(seqs)-1].append(arr[a][1]["App Usage Count"])
    seqs[len(seqs)-1].append(arr[a][1]["User"])
    seqs[len(seqs)-1].append(arr[a][1]["Anomaly"])

seqs = np.array(seqs)

o = open( output_file, 'wt' )

for a in range(len(seqs)):
    for b in range(len(seqs[a])-1):
        o.write(seqs[a][b] + ",")
    o.write(seqs[a][len(seqs[a])-1] + "\n")

o.close()

#out = "tree"
#dot = out + ".dot"
#ng = out + ".png"

#out = tree.export_graphviz(clf, out_file=dot)

#dotfile = StringIO.StringIO()
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


#tree.export_graphviz(clf, out_file=dotfile)



#check_call(['dot','-Tpng',dot,'-o',png])

#(graph,)=pydot.graph_from_dot_data(dotfile.getvalue())
#graph.write_png("dtree.png")


#try:
#    jvm.start()
#    print("Using JRip's Java API to access rules")
#    
#    loader = Loader("weka.core.converters.CSVLoader")
#    labor_data = loader.load_file(os.getcwd() + pathSepD + output_file)
#    labor_data.delete_attribute(30) #Delete User Numbers
#    labor_data.class_is_last()
#    labor_data.randomize(Random(time.time()))
#    
#    
#    jrip = Classifier(classname="weka.classifiers.rules.JRip")
#    jrip.build_classifier(labor_data)
#    rset = jrip.jwrapper.getRuleset()
#    
#    evaluation = Evaluation(labor_data)
#    evaluation.crossvalidate_model(jrip, labor_data, 10, Random(42))
#
#    print(evaluation.summary())
#    print(evaluation.class_details())
#    print(evaluation.matrix())
#    
#    
#    #print("confusionMatrix: " + str(evaluation.confusion_matrix))
#    
#    plot_confusion_matrix(evaluation.confusion_matrix, classes=["Normal", "Abnormal"], normalize=True, title='Normalized confusion matrix')    
#    
#    
#    for i in xrange(rset.size()):
#        r = rset.get(i)        
#        print(str(r.toString(labor_data.class_attribute.jobject)))
#    
#except Exception, e:
#    print(traceback.format_exc())
#finally:
#    jvm.stop()
