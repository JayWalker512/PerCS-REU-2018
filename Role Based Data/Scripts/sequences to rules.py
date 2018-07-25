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


   
    
########################################## Open Existing Sequence ###############################################################    
## Parameters: Range starts at 0


if platform == 'win32':
    pathSep = "\\"
    pathSepD = "\\"
else:
    pathSep = "/"
    pathSepD = "/"
    
input_seq_file = "complete_sequence_flat_96_shift_32.csv"

                  


try:
    jvm.start()
    print("Using JRip's Java API to access rules")
    
    loader = Loader("weka.core.converters.CSVLoader")
    labor_data = loader.load_file(input_seq_file)
    labor_data.delete_attribute(labor_data.num_attributes - 3) #Delete User Numbers
    labor_data.delete_attribute(labor_data.num_attributes - 2) #Delete Attack types
    labor_data.class_is_last()
    labor_data.randomize(Random(time.time()))
    
    
    jrip = Classifier(classname="weka.classifiers.rules.JRip")
    jrip.build_classifier(labor_data)
    rset = jrip.jwrapper.getRuleset()
    
    evaluation = Evaluation(labor_data)
    evaluation.crossvalidate_model(jrip, labor_data, 10, Random(42))

    print(evaluation.summary())
    print(evaluation.class_details())
    print(evaluation.matrix())
    
    classes = []
    
    s = evaluation.matrix().split("\n")
    
    for a in range(len(s)):
        if "=" in s[a] and "|" in s[a]:
            ss = s[a].split("=")
            classes.append(ss[len(ss)-1].lstrip())            
        pass
    

    #print (classes[0])
    
    
    #print("confusionMatrix: " + str(evaluation.confusion_matrix))
    
    plot_confusion_matrix(evaluation.confusion_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')    
    
    
    for i in xrange(rset.size()):
        r = rset.get(i)        
        print(str(r.toString(labor_data.class_attribute.jobject)))
    
except Exception, e:
    print(traceback.format_exc())
finally:
    jvm.stop()
