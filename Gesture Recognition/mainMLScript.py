from numpy import genfromtxt
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

import matplotlib.pyplot as plt

import itertools
import random
import os

import pandas as pd

from pylab import *


import numpy as n
import featureExtraction as featExtr



def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.plot()
    plt.matshow(df_confusion, cmap=cmap) # imshow
    class_names = ['Fist Pump','High Wave','Hand Shake','Fist Bump','Low Wave', 'Point', 'Bandage Wound', 'Blood Pressure', 'Shoulder Radio', 'Motion Over','High-Five','Applause-Clap','Whistling']
    #plt.title(title)
    plt.colorbar()
    tick_marks = n.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, class_names,  rotation=90)


    plt.yticks(tick_marks, class_names)

    width, height = df_confusion.shape
    n.set_printoptions(precision=2)


    for x in xrange(width):
        for y in xrange(height):
            if (x != y):
                plt.annotate(str(df_confusion[x][y]), xy=(x, y), horizontalalignment='center', verticalalignment='center')
            else:
                plt.annotate(str(df_confusion[x][y]), xy=(x, y), horizontalalignment='center', verticalalignment='center', color="white")
    #plt.tight_layout()
    #plt.ylabel(df_confusion.index.name)
    #plt.xlabel(df_confusion.columns.name)
    #savefig("confusion_matrix.png", format="png")

    plt.show()





# File IO
path = 'rawData128SinglePoint.csv'
data = genfromtxt(path, delimiter=',', skip_header=0,usecols=range(0,384))
patientID = genfromtxt(path, delimiter=',', skip_header=0,usecols=[384])

classifications = genfromtxt(path, delimiter=',', skip_header=0,usecols=[385])

patientID = []


# Feature Extraction Script Call
features = featExtr.main(data,classifications)

# Subsetting
features = features[:,:]
n.set_printoptions(threshold='nan')



output_file = "features_" + path
o = open( output_file, 'wt' )


for b in range (len(features[0])):
    o.write(str(features[0][b]) + ", ")
o.write("\n")

o.close()



# Cross- Validation Schemes
#This works by splitting the data into some number of consecutive training and test sets
kf = KFold(len(classifications), 10, shuffle=True)
loso = LeaveOneLabelOut(patientID)

#print(loso)

# Classifier Selection
#clf = LogisticRegression(C=10,penalty='l1', tol=.01)

#clf = KNeighborsClassifier(n_neighbors=3)
#clf = svm.SVC(kernel="linear")
#clf = tree.DecisionTreeClassifier()
#clf = GaussianNB()
clf = RandomForestClassifier(max_depth=35, n_estimators=1000, max_features=15)
#clf = AdaBoostClassifier(n_estimators=1000)



# Needed for RBF SVM
#features = preprocessing.scale(features)



#confMat = n.zeros((9,9))
allPredictions = []
allClassifications = []

#print (classifications)


for train, test in kf:
    clf.fit(features[train], classifications[train])
    predictions = clf.predict(features[test])
    #confMat = confMat + confusion_matrix(classifications[test],predictions)
    allPredictions.append( predictions)
    allClassifications.append(classifications[test])



allPredictions = n.concatenate(allPredictions)
allClassifications = n.concatenate(allClassifications)



print(classification_report(allClassifications,allPredictions))
#print(confMat)
print("Accuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))



## Compute confusion matrix



#==============================================================================
# allClassifications = []
# allPredictions = []
#
# for i in range(650):
#     allClassifications.append(i//50)
#     allPredictions.append(i//50)
#
# allPredictions[150] = 3
# allPredictions[151] = 3
# allPredictions[152] = 3
#
# allPredictions[250] = 12
# allPredictions[251] = 12
# allPredictions[252] = 12
# allPredictions[253] = 12
# allPredictions[254] = 12
# allPredictions[255] = 12
# allPredictions[256] = 12
# allPredictions[257] = 12
#
# allPredictions[300] = 2
# allPredictions[301] = 2
# allPredictions[302] = 2
# allPredictions[303] = 2
# allPredictions[304] = 2
# allPredictions[305] = 2
# allPredictions[306] = 2
# allPredictions[307] = 2
#
# allPredictions[350] = 12
# allPredictions[351] = 12
# allPredictions[352] = 12
# allPredictions[353] = 12
# allPredictions[354] = 12
# allPredictions[355] = 12
# allPredictions[356] = 5
# allPredictions[357] = 5
# allPredictions[358] = 5
# allPredictions[359] = 5
#
# allPredictions[400] = 5
# allPredictions[401] = 5
# allPredictions[402] = 5
# allPredictions[403] = 5
# allPredictions[404] = 5
# allPredictions[405] = 5
#
# allPredictions[500] = 3
# allPredictions[501] = 3
# allPredictions[502] = 3
# allPredictions[503] = 3
#==============================================================================




#print (allClassifications)





cnf_matrix = confusion_matrix(allClassifications,allPredictions)
print(cnf_matrix)
#print (cnf_matrix)


y_actu = pd.Series(allClassifications, name='Actual')
y_pred = pd.Series(allPredictions, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_conf_norm = n.around(df_conf_norm, decimals=4)
