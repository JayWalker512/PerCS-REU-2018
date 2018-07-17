from __future__ import print_function

from sys import platform

import csv
import numpy as np

import random



def generateNormal(amount = 25, walking = "0", p = .5):
    global X
    global Z
    global min_time
    global max_time
    global act_duration_mins
    global act_duration_maxs
    global act_app_usages
     
    
    max_user = -1 
    if (p > 1):
        p = 1
    if (p < 0):
        p = 0
    for a in range(1,len(Z)):
        max_user = max(max_user, int(Z[a][0]))
    
    a = 0
    while (a < amount):
        X.append([])
        Z.append([])
        Z[len(Z)-1].append(str(max_user + 1))
        Z[len(Z)-1].append("Normal")
        
        duration = 0
        app_usage = 0
        delete = True
        
        c_time = random.randint(min_time, max_time)
        
        for b in range(len(X[0])-3):
            if ("Start" in X[0][b]):
                if (float(X[len(X)-1][b-1]) > 0):
                    X[len(X)-1].append(str(int(c_time)))
                else:
                    X[len(X)-1].append("0")
            elif ("End" in X[0][b]):
                if (float(X[len(X)-1][b-1]) > 0):
                    s = X[0][b].split()
                    act = s[0]
                    if (len(s) > 2):
                        for c in range(1, len(s)-1):
                            act = act + " " + s[c]                        
                    min_t = act_duration_mins[act + " Delta Min"]
                    max_t = act_duration_maxs[act + " Delta Max"]
                    duration = random.uniform(min_t, max_t)
                    c_time = c_time + (duration * 1000) + random.randint(1000, 3000)
                    X[len(X)-1].append(str(int(X[len(X)-1][b-1]) + (duration * 1000)))
                else:
                    X[len(X)-1].append("0")
            elif ("Delta" in X[0][b]):
                if (float(X[len(X)-1][b-1]) > 0):                    
                    X[len(X)-1].append(str(duration))
                    duration = 0
                else:
                    X[len(X)-1].append("0")
            else:
                if (random.random() < p):
                    if (X[0][b].lstrip() in act_app_usages):
                        #app_usage = app_usage + 1
                        X[len(X)-1].append("0")
                    else:
                        delete = False
                        X[len(X)-1].append("1")
                        
                else:
                    X[len(X)-1].append("0")
        
        if ("0" in walking): #if not walking
            X[len(X)-1].append(str(random.uniform(0.0, 3.0)))
            X[len(X)-1].append("0")
        else:
            seq_start = 4687389847 # Some Date in 2118
            seq_end = -1
            for b in range(len(X[0])-3):
                if ("Start" in X[0][b]):
                    if (float(X[len(X)-1][b]) > 0):
                        seq_start = min(seq_start, float(X[len(X)-1][b]))
                if ("End" in X[0][b]):
                    seq_end = max(seq_end, float(X[len(X)-1][b]))
            if (seq_end < 0):
                seq_start = 0 # Some Date in 2118
                seq_end = 0
            X[len(X)-1].append(str((float(seq_end) - float(seq_start)) * (1.39/1000.0)))
            X[len(X)-1].append("1")
        
        X[len(X)-1].append(str(app_usage))
        
        if (delete):
            X.pop(len(X)-1)
        else:
            a = a + 1

def generateAttackSpoofingGPS(amount = 25, p = .5, attackType = "StationaryDrift"):
    global X
    global Z

    ind = X[0].index("Distance Magnitude")
    rate = 0
    
    if ("StationaryDrift" in attackType):
        generateNormal(amount = amount, walking = "0", p = p)
        rate = (.695/1000.0)
            
    if ("WalkingDrift" in attackType):
        generateNormal(amount = amount, walking = "1", p = p)
        rate = (2.78/1000.0)
        
    for a in range(len(X)-1,len(X)-1-amount,-1):
        Z[a][len(Z[a])-1] = "Abnormal"
        seq_start = 4687389847 # Some Date in 2118
        seq_end = -1
        for b in range(len(X[0])-3):
            if ("Start" in X[0][b]):
                if (float(X[a][b]) > 0):
                    seq_start = min(seq_start, float(X[a][b]))
            if ("End" in X[0][b]):
                seq_end = max(seq_end, float(X[a][b]))
        X[a][ind] = str((float(seq_end) - float(seq_start)) * rate)

        

########################################## Open Existing Sequence ###############################################################    
## Parameters: Range starts at 0

if platform == 'win32':
    pathSep = "\\"
    pathSepD = "\\"
else:
    pathSep = "/"
    pathSepD = "/"
    
input_seq_file = "sequence_flat_96_shift_32.csv"

                  
result=np.array(list(csv.reader((open(input_seq_file, "rb")), delimiter=",")))

X = np.delete(result, np.s_[len(result[0])-2:len(result[0])], 1)
#Y = np.delete(result, np.s_[0:len(result[0])-5], 1)
#Y = np.delete(Y, np.s_[len(Y[0])-2:len(Y[0])], 1)
Z = np.delete(result, np.s_[0:len(result[0])-2], 1)

min_time = 4687389847
max_time = -1
act_duration_mins = {}
act_duration_maxs = {}
act_app_usages = []

thresh = 2

for a in range(1,len(X)):
    for b in range (len(X[0])):
        if ("Start" in X[0][b] and float(X[a][b]) > 1000):
            min_time = min(min_time, float(X[a][b]))
        elif ("End" in X[0][b]):
            max_time = max(max_time, float(X[a][b]))
        elif ("Delta" in X[0][b]):
            delta = X[0][b]
            if (delta + " Min" not in act_duration_mins):                
                act_duration_maxs[delta + " Max"] = float(X[a][b])
                act_duration_mins[delta + " Min"] = float(X[a][b])
                if (float(X[a][b]) < thresh):
                    act_duration_mins[delta + " Min"] = 100
                    
            else:
                act_duration_maxs[delta + " Max"] = max(float(X[a][b]), act_duration_maxs[delta + " Max"])
                if (float(X[a][b]) > thresh):
                    act_duration_mins[delta + " Min"] = min(float(X[a][b]), act_duration_mins[delta + " Min"])
        else:
            label = X[0][b].lstrip()
            if (label not in act_app_usages):
                add = True
                if (label + " Delta Min" not in list(act_duration_mins)):
                    for c in range (b+1, len(X[0])):
                        if (label + " Start" in X[0][c]):
                            add = False
                else:
                    add = False
                
                if (add):
                    act_app_usages.append(X[0][b].lstrip())
                    
X = X.tolist()
Z = Z.tolist()                 
##################################### Inject Attacks Here ############################

generateNormal()
generateNormal(walking = "1")

generateAttackSpoofingGPS(attackType = "StationaryDrift")
generateAttackSpoofingGPS(attackType = "WalkingDrift")



     
##################### DOS Attack ############################
#    elif ("500101" in arr[a][len(arr[0])-2]):
#        arr[a][29] = str(random.randint(5, 10))
#        arr[a][len(arr[0])-1] = "Abnormal"   

##################################### Write New File #################################
Y = []
Y.append([])

output_file = "complete_" + input_seq_file

o = open( output_file, 'wt' )

for a in range(len(X)):
    for b in range(len(X[0])):
        o.write(X[a][b] + ",")
    for b in range(len(Y[0])):
        o.write(Y[a][b] + ",")
    for b in range(len(Z[0])-1):
        o.write(Z[a][b] + ",")
    o.write(Z[a][len(Z[a])-1] + "\n")

o.close()
