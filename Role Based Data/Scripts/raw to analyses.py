from common import *

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