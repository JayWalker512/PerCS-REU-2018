from __future__ import print_function


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
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    np.set_printoptions(precision=2)
    cm = np.around(cm, decimals=2)
    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],                 horizontalalignment="center",                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


        



########################################## Window Extractor! ###############################################################    
## Parameters: Range starts at 0
col_x = 1
col_y = 2
col_z = 3
col_user = 9
col_class = 12
win_start = 1
#user_start = 100104
#user_end = 100111

## Look into variable these
window_size = 128
shift_amount = 64

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
        output_file = "output\win_" + str(a) + "_shift_" + str(b) + ".csv"       
        o = open( output_file, 'wt' )
        
        for input_file in glob.glob('*.csv'): 
            print (input_file)
                       
            result=np.array(list(csv.reader((open(input_file, "rb")), delimiter=",")))
            ##if (str(100106) in result[1][col_user] or str(100107) in result[1][col_user] or str(100109) in result[1][col_user]):           
            
            ##print result
            twin_start = win_start              
            
            while((twin_start + a) <= len(result)):
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

#print (result)

out, labels, users = calculateFeatures(result)
output_file = "output\\features_" + "win_" + str(a) + "_shift_" + str(b) + ".csv"
input_file =  "features_" + "win_" + str(a) + "_shift_" + str(b) + ".csv"
o = open( output_file, 'wt' )
    
for a in range(len(out)):
    for b in range (len(out[a])):
        o.write(str(out[a][b]) + ", ")
    o.write(str(labels[a]) + ", " + str(users[a]) + "\n")   

o.close()
 

print ("Beginning Feature Analysis! This may take a few minutes!") 




############################### Cop K-means ###################

#input_file = "features_win_128_shift_64.csv" ################# INSERT FEATURE EXTRACTED CSV FILE HERE!!!!! ######
input_folder = "output" ################# INSERT FEATURE EXTRACTED CSV FILE HERE!!!!! ######
print ("************************")
print (input_file)
org = input_file.rpartition('.')[0]

#input_file = org
X=np.array(list(csv.reader((line.replace('\0','') for line in open(os.getcwd() + "\\" + input_folder + "\\" + input_file, "rb")), delimiter="," )))

X1 = np.array(list([X[X[:,len(X[0])-2]==k] for k in np.unique(X[:,len(X[0])-2])]))


    
#X1 = np.delete(X1, len(X1[0][0])-1, 2)
#X1 = np.delete(X1, len(X1[0][0])-1, 2)
#X1 = X1.astype(np.float)
#X1 = preprocessing.scale(X1)

#print (X1)
    
true_user = []

n_cluster_eyes = 1
n_cluster_flashlight = 5
n_cluster_high_wave = 1
n_cluster_shoulder_radio = 6
n_cluster_take_bp = 4
n_cluster_whistle = 6
n_cluster_transition = 5
n_cluster_none = 1

n_max_activities = 40

X_eyes = []
Y_eyes = []

X_flash = []
Y_flash = []

X_wave = []
Y_wave = []

X_radio = []
Y_radio = []

X_bp = []
Y_bp = []

X_whistle = []
Y_whistle = []

X_transition = []
Y_transition = []

X_none = []
Y_none = []

for a in range(len(X1)):
#==============================================================================
#     if ('Eyes' in X1[a][0][len(X1[a][0])-2]):
#         X_eyes = X1[a]
#         
# #==============================================================================
# #         if (len(X_eyes) > n_max_activities):
# #             np.random.shuffle(X_eyes)
# #             X_eyes = X_eyes[0:n_max_activities]   
# #==============================================================================
#         
#         true_user.append(X_eyes[:,len(X_eyes[0])-1])
#         X_eyes = np.delete(X_eyes, len(X_eyes[0])-1, 1)
#         X_eyes = np.delete(X_eyes, len(X_eyes[0])-1, 1)
#         X_eyes = X_eyes.astype(np.float)
#         X_eyes = preprocessing.scale(X_eyes)
#         Y_eyes = KMeans(n_clusters=n_cluster_eyes, random_state=0).fit(X_eyes).predict(X_eyes)
#         pass
#==============================================================================
    if ('Flash' in X1[a][0][len(X1[a][0])-2]):
        X_flash = X1[a]
        
#==============================================================================
#         if (len(X_flash) > n_max_activities):            
#             np.random.shuffle(X_flash)            
#             X_flash = X_flash[0:n_max_activities]   
#==============================================================================
            
        true_user.append(X_flash[:,len(X_flash[0])-1])
        X_flash = np.delete(X_flash, len(X_flash[0])-1, 1)
        X_flash = np.delete(X_flash, len(X_flash[0])-1, 1)
        X_flash = X_flash.astype(np.float)
        X_flash = preprocessing.scale(X_flash)
        Y_flash = KMeans(n_clusters=n_cluster_flashlight, random_state=0).fit(X_flash).predict(X_flash)
        pass
#==============================================================================
#     if ('Wave' in X1[a][0][len(X1[a][0])-2]):
#         X_wave = X1[a]
#         
# #==============================================================================
# #         if (len(X_wave) > n_max_activities):
# #             np.random.shuffle(X_wave)
# #             X_wave = X_wave[0:n_max_activities]   
# #==============================================================================
#             
#         true_user.append(X_wave[:,len(X_wave[0])-1])
#         X_wave = np.delete(X_wave, len(X_wave[0])-1, 1)
#         X_wave = np.delete(X_wave, len(X_wave[0])-1, 1)
#         X_wave = X_wave.astype(np.float)
#         X_wave = preprocessing.scale(X_wave)
#         Y_wave = KMeans(n_clusters=n_cluster_high_wave, random_state=0).fit(X_wave).predict(X_wave)
#         pass
#==============================================================================
    if ('Radio' in X1[a][0][len(X1[a][0])-2]):
        X_radio = X1[a]
        
#==============================================================================
#         if (len(X_radio) > n_max_activities):
#             np.random.shuffle(X_radio)
#             X_radio = X_radio[0:n_max_activities]   
#==============================================================================
            
            
        true_user.append(X_radio[:,len(X_radio[0])-1])
        X_radio = np.delete(X_radio, len(X_radio[0])-1, 1)
        X_radio = np.delete(X_radio, len(X_radio[0])-1, 1)
        X_radio = X_radio.astype(np.float)
        X_radio = preprocessing.scale(X_radio)
        Y_radio = KMeans(n_clusters=n_cluster_shoulder_radio, random_state=0).fit(X_radio).predict(X_radio)
        pass
    if ('BP' in X1[a][0][len(X1[a][0])-2]):
        X_bp = X1[a]
        
#==============================================================================
#         if (len(X_bp) > n_max_activities):
#             np.random.shuffle(X_bp)
#             X_bp = X_bp[0:n_max_activities]   
#==============================================================================
            
            
        true_user.append(X_bp[:,len(X_bp[0])-1])
        X_bp = np.delete(X_bp, len(X_bp[0])-1, 1)
        X_bp = np.delete(X_bp, len(X_bp[0])-1, 1)
        X_bp = X_bp.astype(np.float)
        X_bp = preprocessing.scale(X_bp)
        Y_bp = KMeans(n_clusters=n_cluster_take_bp, random_state=0).fit(X_bp).predict(X_bp)
        pass
    if ('Whistle' in X1[a][0][len(X1[a][0])-2]):
        X_whistle = X1[a]
        
#==============================================================================
#         if (len(X_whistle) > n_max_activities):
#             np.random.shuffle(X_whistle)
#             X_whistle = X_whistle[0:n_max_activities]   
#==============================================================================
            
            
        true_user.append(X_whistle[:,len(X_whistle[0])-1])
        X_whistle = np.delete(X_whistle, len(X_whistle[0])-1, 1)
        X_whistle = np.delete(X_whistle, len(X_whistle[0])-1, 1)
        X_whistle = X_whistle.astype(np.float)
        X_whistle = preprocessing.scale(X_whistle)
        Y_whistle = KMeans(n_clusters=n_cluster_whistle, random_state=0).fit(X_whistle).predict(X_whistle)
        pass
#    if ('None' in X1[a][0][len(X1[a][0])-2]):
#        X_none = X1[a]

#        if (len(X_none) > n_max_activities):
#           np.random.shuffle(X_none)
#           X_none = X_none[0:n_max_activities]   
            
            
#        true_user.append(X_none[:,len(X_none[0])-1])
#        X_none = np.delete(X_none, len(X_none[0])-1, 1)
#        X_none = np.delete(X_none, len(X_none[0])-1, 1)
#        X_none = X_none.astype(np.float)
#        X_none = preprocessing.scale(X_none)
#        Y_none = KMeans(n_clusters=n_cluster_none, random_state=0).fit(X_none).predict(X_none)
#        pass
    if ('Transition' in X1[a][0][len(X1[a][0])-2]):
        X_transition = X1[a]
        
#==============================================================================
#         if (len(X_transition) > n_max_activities):
#             np.random.shuffle(X_transition)
#             X_transition = X_transition[0:n_max_activities]   
#==============================================================================
            
        true_user.append(X_transition[:,len(X_transition[0])-1])
        X_transition = np.delete(X_transition, len(X_transition[0])-1, 1)
        X_transition = np.delete(X_transition, len(X_transition[0])-1, 1)
        X_transition = X_transition.astype(np.float)
        X_transition = preprocessing.scale(X_transition)
        Y_transition = KMeans(n_clusters=n_cluster_transition, random_state=0).fit(X_transition).predict(X_transition)
        pass

X = []
Y = []

if (len(Y_eyes) > 0):
    Y_eyes = map(str, Y_eyes)
    for a in range(len(Y_eyes)):
        Y_eyes[a] = "Examine Eyes " + str(Y_eyes[a])
    X.append(X_eyes)
    Y.append(Y_eyes)

if (len(Y_flash) > 0):
    Y_flash = map(str, Y_flash)
    for a in range(len(Y_flash)):
        Y_flash[a] = "Flashlight " + str(Y_flash[a])
    X.append(X_flash)
    Y.append(Y_flash)

if (len(Y_wave) > 0):
    Y_wave = map(str, Y_wave)
    for a in range(len(Y_wave)):
        Y_wave[a] = "High Wave " + str(Y_wave[a])
    X.append(X_wave)
    Y.append(Y_wave)

if (len(Y_radio) > 0):
    Y_radio = map(str, Y_radio)
    for a in range(len(Y_radio)):
        Y_radio[a] = "Shoulder Radio " + str(Y_radio[a])
    X.append(X_radio)
    Y.append(Y_radio)

if (len(Y_bp) > 0):
    Y_bp = map(str, Y_bp)
    for a in range(len(Y_bp)):
        Y_bp[a] = "Take BP " + str(Y_bp[a])
    X.append(X_bp)
    Y.append(Y_bp)

if (len(Y_whistle) > 0):
    Y_whistle = map(str, Y_whistle)
    for a in range(len(Y_whistle)):
        Y_whistle[a] = "Whistle " + str(Y_whistle[a])
    X.append(X_whistle)
    Y.append(Y_whistle)
    
if (len(Y_transition) > 0):
    Y_transition = map(str, Y_transition)
    for a in range(len(Y_transition)):
        Y_transition[a] = "Transition " + str(Y_transition[a])
    X.append(X_transition)
    Y.append(Y_transition)
    
if (len(Y_none) > 0):
    Y_none = map(str, Y_none)
    for a in range(len(Y_none)):
        Y_none[a] = "None " + str(Y_none[a])
    X.append(X_none)
    Y.append(Y_none)

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

if (len(X2) > 1):
    X = np.vstack((X2[0],X2[1]))
for a in range(2, len(X2)):
    X = np.vstack((X,X2[a]))

Y = [item for sublist in Y for item in sublist]
Y = np.array(Y)

true_user = [item for sublist in true_user for item in sublist]
true_user = np.array(true_user)



X = np.column_stack((X,Y))
X = np.column_stack((X,true_user))

for a in range(3):
    np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X)

true_labels = X[:,len(X[0])-2]
true_user = X[:,len(X[0])-1]
X = np.delete(X, len(X[0])-1, 1)
X = np.delete(X, len(X[0])-1, 1)
X = X.astype(np.float)
#X = preprocessing.scale(X)


    
# Cross- Validation Schemes
kf = KFold(len(true_labels), 10, shuffle=True)
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
    if ('None' in allPredictions[a]):
        allPredictions[a] = 'None'
        pass
    if ('Eyes' in allPredictions[a]):
        allPredictions[a] = 'Examine Eyes'
        pass
    if ('Flash' in allPredictions[a]):
        allPredictions[a] = 'Flashlight'
        pass
    if ('Wave' in allPredictions[a]):
        allPredictions[a] = 'High Wave'
        pass
    if ('Radio' in allPredictions[a]):
        allPredictions[a] = 'Shoulder Radio'
        pass
    if ('BP' in allPredictions[a]):
        allPredictions[a] = 'Take BP'
        pass
    if ('Whistle' in allPredictions[a]):
        allPredictions[a] = 'Whistle'
        pass
    if ('Transition' in allPredictions[a]):
        allPredictions[a] = 'Transition'
        pass
    if ('None' in allPredictions[a]):
        allPredictions[a] = 'None'
        pass

for a in range(len(allClassifications)):
    if ('None' in allClassifications[a]):
        allClassifications[a] = 'None'
        pass
    if ('Eyes' in allClassifications[a]):
        allClassifications[a] = 'Examine Eyes'
        pass
    if ('Flash' in allClassifications[a]):
        allClassifications[a] = 'Flashlight'
        pass
    if ('Wave' in allClassifications[a]):
        allClassifications[a] = 'High Wave'
        pass
    if ('Radio' in allClassifications[a]):
        allClassifications[a] = 'Shoulder Radio'
        pass
    if ('BP' in allClassifications[a]):
        allClassifications[a] = 'Take BP'
        pass
    if ('Whistle' in allClassifications[a]):
        allClassifications[a] = 'Whistle'
        pass
    if ('Transition' in allClassifications[a]):
        allClassifications[a] = 'Transition'
        pass
    if ('None' in allClassifications[a]):
        allClassifications[a] = 'None'
        pass
  

pred_valid = allPredictions
   
print ('Get Results')

print(classification_report(allClassifications,allPredictions))

print("Accuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))


cnf_matrix = confusion_matrix(allClassifications,allPredictions)

np.set_printoptions(threshold='nan')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[], normalize=True,                      title='Normalized confusion matrix')
