from __future__ import print_function

from sys import platform

import csv
import numpy as np

import random


#amount: how many new examples to generate
#whether the the "walking" feature should == 1 in new examples
#p: probability of app being used in a particular example
def generateNormal(amount = 25, walking = "0", p = .5):
    global X
    global Z
    global min_time
    global max_time
    global act_duration_mins
    global act_duration_maxs
    global act_app_usages
    
    global bool_y
    global rate_y
    
    ind = X[0].index("Distance Magnitude")
     
    
    max_user = -1 
    p = np.clip(p, 0, 1) #restrict p to range [0,1]
    for a in range(1,len(Z)):
        max_user = max(max_user, int(Z[a][0])) #get the max current UserID, we'll increment this as we add examples
    
    a = 0 #this is just an index variable we count up until arriving at 'amount'
    while (a < amount):
        X.append([])
        Y.append([])
        
        #This loop never runs! len(Y[0]) == 0
        for b in range(len(Y[0])):
            label = Y[0][b]
            if (label in bool_y):
                if (random.random() < p):                    
                    Y[len(Y)-1].append("0")
                else:                    
                    Y[len(Y)-1].append("1")
            elif ("GPS Rate" not in label):                                 
                rate = rate_y[label]
                Y[len(Y)-1].append(str(random.uniform(rate * .9, rate * 1.1)))
            else:
                Y[len(Y)-1].append("0")
        
        
        Z.append([])
        Z[len(Z)-1].append(str(max_user + 1)) #for the data we're adding, increment user id
        Z[len(Z)-1].append("Normal")
        Z[len(Z)-1].append("Normal")
        
        duration = 0
        app_usage = 0
        delete = True
        
        c_time = random.randint(min_time, max_time) #"current" time?
        
        #this loop generates new start & end times for the various activities, and calculates the elapsed time.
        #These new values are put in columns of X for the current example 'a'
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
                
            rate = (1.39/1000.0) #scaled GPS magnitude accumulation rate            
            r2 = random.uniform(rate * .9, rate * 1.1)
            X[len(X)-1].append(str((float(seq_end) - float(seq_start)) * r2))                
            X[len(X)-1].append("1")
        
        if ("GPS Rate" in ind_y):
            seq_start = 4687389847 # Some Date in 2118
            seq_end = -1
            for b in range(len(X[0])-3):
                if ("Start" in X[0][b]):
                    if (float(X[len(X)-1][b]) > 0):
                        seq_start = min(seq_start, float(X[len(X)-1][b]))
                if ("End" in X[0][b]):
                    seq_end = max(seq_end, float(X[len(X)-1][b]))
            
            rate = float(X[len(X)-1][ind]) / (seq_end - seq_start)
            Y[len(X)-1][ind_y["GPS Rate"]] = (str(rate))
        
        X[len(X)-1].append(str(app_usage))
        
        if (delete):
            X.pop(len(X)-1)
            Y.pop(len(Y)-1)
            Z.pop(len(Z)-1)
        else:
            a = a + 1

#amount: how many examples to generate
#p: probability of "Device Rooted" and "Mock GPS"
def generateAttackSpoofingGPS(amount = 25, p = .5, attackType = "StationaryDrift"):
    global X
    
    #In this context, Y is for new columns we're adding I guess?
    global Y
    global Z
    
    global ind_y
    global bool_y
    global rate_y

    index_of_distance_magnitude = X[0].index("Distance Magnitude")
    rate = 0           
    
    if ("Mock GPS" not in ind_y):
        Y[0].append("Mock GPS")
        ind_y["Mock GPS"] = len(Y[0])-1 #note the index of the "Mock GPS" column
        bool_y.append("Mock GPS")
        
        #apply 1 or 0 in "Mock GPS" column according to probability p
        for b in range(1, len(Y)):
            if (random.random() < p):
                Y[b].append("1")                        
            else:
                Y[b].append("0")
        
    if ("Device Rooted" not in ind_y):
        Y[0].append("Device Rooted")
        ind_y["Device Rooted"] = len(Y[0])-1 #note the index of the "Mock GPS" column
        bool_y.append("Device Rooted")
        
        #apply 1 or 0 in "Device Rooted" column according to probability p
        for b in range(1, len(Y)):
            if (random.random() < p):
                Y[b].append("1")                        
            else:
                Y[b].append("0")
    
    
    if ("GPS Rate" not in ind_y):
        Y[0].append("GPS Rate")
        ind_y["GPS Rate"] = len(Y[0])-1 #note the index of the "GPS Rate" column
        rate_y["GPS Rate"] = .0014    
    
        #iterate over each example and add the new "GPS Rate" column and update  GPS Distance Magnitude
        for a in range(1, len(Y)):
            seq_start = 4687389847 # Some Date in 2118
            seq_end = -1
            for b in range(len(X[0])-3):
                if ("Start" in X[0][b]):
                    if (float(X[a][b]) > 0):
                        seq_start = min(seq_start, float(X[a][b]))
                if ("End" in X[0][b]):
                    seq_end = max(seq_end, float(X[a][b]))
                    
            #determine GPS rate by assuming constant rate over sequence duration
            rate = float(X[a][index_of_distance_magnitude]) / (seq_end - seq_start)   
            #add some random variability
            r2 = random.uniform(rate * .9, rate * 1.1)
            #calculate new distance
            X[a][index_of_distance_magnitude] = str((float(seq_end) - float(seq_start)) * r2)            
            #insert the rate to features for this example
            Y[a].append(str(r2))    
    
    #if we're not walking, add a small drift
    if ("StationaryDrift" in attackType):
        generateNormal(amount = amount, walking = "0", p = p)
        rate = (.695/1000.0)
            
    #if we're walking, need to account for walking GPS rate + drift
    if ("WalkingDrift" in attackType):
        generateNormal(amount = amount, walking = "1", p = p)
        rate = (2.78/1000.0)         
        
    #randomly decide if "Mock GPS" and "Device Rooted" are set.
    for b in range(len(Y)-1,len(Y)-1-amount,-1):
        r = random.randint(0,2)
        
        #first and last cases are identical? Why?
        if (r == 0):
            Y[b][ind_y["Mock GPS"]] = "1"
            Y[b][ind_y["Device Rooted"]] = "1"
        elif (r == 1):
            Y[b][ind_y["Device Rooted"]] = "1"
            Y[b][ind_y["Mock GPS"]] = "0"
        else:
            Y[b][ind_y["Mock GPS"]] = "1"
            Y[b][ind_y["Device Rooted"]] = "1"
        
    #iterate (backwards) through the examples we've added and label them appropriately    
    for a in range(len(X)-1,len(X)-1-amount,-1):
        Z[a][len(Z[a])-2] = "GPS Spoofing"
        Z[a][len(Z[a])-1] = "Abnormal"
        seq_start = 4687389847 # Some Date in 2118
        seq_end = -1
        
        for b in range(len(X[0])-3):
            if ("Start" in X[0][b]):
                if (float(X[a][b]) > 0):
                    seq_start = min(seq_start, float(X[a][b]))
            if ("End" in X[0][b]):
                seq_end = max(seq_end, float(X[a][b]))
                
        #update the GPS Rate and Distance Magnitude for the rates we chose earlier
        r2 = random.uniform(rate * .9, rate * 1.1)
        Y[a][ind_y["GPS Rate"]] = str(r2) 
        X[a][index_of_distance_magnitude] = str((float(seq_end) - float(seq_start)) * r2)        
        

def generateAttackBatteryDrain(amount = 25, p = .5, attackType = "Standard"):
    global X
    global Y
    global Z
    
    global ind_y
    global bool_y
    global rate_y
    
    rate = (.53171/60)
 
    #add label to the new column
    if ("Energy Rate" not in ind_y):
        Y[0].append("Energy Rate")
        ind_y["Energy Rate"] = len(Y[0])-1
        rate_y["Energy Rate"] = rate
        
#        Y[0].append("Energy Magnitude")
#        ind_y["Energy Magnitude"] = len(Y[0])-1    
#        rate_y["Energy Magnitude"] = 0
    
        for a in range(1, len(Y)):
            seq_start = 4687389847 # Some Date in 2118
            seq_end = -1
            
            for b in range(len(X[0])-3):
                if ("Start" in X[0][b]):
                    if (float(X[a][b]) > 0):
                        seq_start = min(seq_start, float(X[a][b]))
                if ("End" in X[0][b]):
                    seq_end = max(seq_end, float(X[a][b]))
            
            r2 = random.uniform(rate * .9, rate * 1.1)
            Y[a].append(str(r2))    
            #Y[a].append(str(int(((seq_end - seq_start)/1000.0) * r2)))
    
    generateNormal(amount = int(amount/2), walking = "0", p = p)
    generateNormal(amount = amount - int(amount/2), walking = "1", p = p)
    
    if ("Standard" in attackType):        
        rate = .00433027 #arbitrary rate?
            
    #doesn't seem to make a difference!
    if ("SleepDeprivation" in attackType):        
        rate = .00433027 #arbitrary rate?
        
    #iterate backwards and add labels for attack type
    for a in range(len(X)-1,len(X)-1-amount,-1):
        Z[a][len(Z[a])-2] = "Battery Drain"
        Z[a][len(Z[a])-1] = "Abnormal"
        seq_start = 4687389847 # Some Date in 2118
        seq_end = -1
        for b in range(len(X[0])-3):
            if ("Start" in X[0][b]):
                if (float(X[a][b]) > 0):
                    seq_start = min(seq_start, float(X[a][b]))
            if ("End" in X[0][b]):
                seq_end = max(seq_end, float(X[a][b]))
                
        #add battery drain rate to the appropriate column with some noise
        r2 = random.uniform(rate * .9, rate * 1.1)
        Y[a][ind_y["Energy Rate"]] = (str(r2))    
        #Y[a][ind_y["Energy Rate"]] = (str(int(((seq_end - seq_start)/1000.0) * r2)))

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
        if ("Walking" in X[0][b] and "0" in X[a][b]):
            X[a][b-1] = str(random.uniform(0.0, 3.0))
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

Y = []
for a in range(len(X)):
    Y.append([])
    if(a > 0):
        Z[a].append("Normal")
    else:
        Z[a].append("Classification")

ind_y = {}
bool_y = []
rate_y = {}

#X has training data, including Shoulder Radio, Shoulder Radio Start,...,through Walking, App Usage Count
#Y is empty at this point 
#Z has User ID, Anomaly type, and classification of "Normal" or "Anomaly"

##################################### Inject Attacks Here ############################

generateNormal(amount = 40)
generateNormal(amount = 15, walking = "1")
generateNormal(amount = 50)

generateAttackSpoofingGPS(attackType = "StationaryDrift")
generateAttackSpoofingGPS(attackType = "WalkingDrift")

generateAttackBatteryDrain(attackType = "Standard")
generateAttackBatteryDrain(attackType = "Standard")



     
##################### DOS Attack ############################
#    elif ("500101" in arr[a][len(arr[0])-2]):
#        arr[a][29] = str(random.randint(5, 10))
#        arr[a][len(arr[0])-1] = "Abnormal"   

##################################### Write New File #################################


output_file = "complete_" + input_seq_file

o = open( output_file, 'wt' )

for a in range(len(X)):
    for b in range(len(X[0])):
        if ("Start" not in X[0][b] and "End" not in X[0][b]):
            o.write(X[a][b] + ",")
    for b in range(len(Y[0])):
        if ("Start" not in Y[0][b] and "End" not in Y[0][b]):
            o.write(Y[a][b] + ",")
    for b in range(len(Z[0])-1):
        if ("Start" not in Z[0][b] and "End" not in Z[0][b]):
            o.write(Z[a][b] + ",")
    o.write(Z[a][len(Z[a])-1] + "\n")

o.close()
