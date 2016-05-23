#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

#----------------------------------------------------
#----------------------------------------------------

### Task 1: I first make a complete list of features..  I later find the most 
# important features for use in my machine learning algorithm

target_label = ['poi']
email_features_list = [
    # 'email_address', # remit email address; informational label
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages'
    ]
    
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = target_label + financial_features_list + email_features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Exploratory Data Analysis

from collections import defaultdict
check_total = defaultdict(list)

plt.close('all')
# Create box plots to look at univariate data distribution and outliers
for feature in features_list[1:]:
    
    if feature=='email_address': #skip the email address.. no numerical vals
        continue
        
    gather_data=[]
    for person in data_dict.keys():
        try:        
            val=float(data_dict[person][feature])
            gather_data.append(val)
        except:
            continue
    gather_data_np=np.array(gather_data)
    gather_data_np=gather_data_np[~np.isnan(gather_data_np)]
    
    check_total[feature]=np.sum(gather_data_np)
    
    plt.figure()    
    plt.boxplot(gather_data_np)
    plt.title(feature)
    plt.ylabel('Value')
    
# Fix a couple of records so check_total matches enron61702insiderpay.pdf
    
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'

data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

### Task 2: Remove outliers / non-critical entries
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E',0) # all NaN

#----------------------------------------------------
#----------------------------------------------------

#Gather some basic information about the dataset now that outliers and 
#extraneous entries are removed

#Number of Records in the dataset
print "Number of records in the dataset = ", str(len(data_dict.keys()))

good_count=0
bad_count=0
for person in data_dict.keys():
    for feature in features_list:
        if data_dict[person][feature]!='NaN':
            good_count+=1 #1581
        else:
            bad_count+=1 #1299

print "Percent of data that is non-NaN: ", 
float(good_count)/float(good_count+bad_count) #.5489

#Number of POI's (18)
counter = 0
for person in data_dict.keys():
    if data_dict[person]['poi']==True:
        counter+=1
print "Number of POIs = ", str(counter)

#----------------------------------------------------
#----------------------------------------------------

## Create percent data coverage checker function
def percent_count_coverage(feature_name, input_data):
     
     total_count=0 
     total_poi=0
     total_nonpoi=0
     
     full_counter = 0
     poi_counter=0
     nonpoi_counter=0
	
     for person in input_data.keys():
         if data_dict[person]['poi']==1:
             if data_dict[person][feature_name]!='NaN':
                 poi_counter+=1
             total_poi+=1
         elif data_dict[person]['poi']==0:
             if data_dict[person][feature_name]!='NaN':               
                 nonpoi_counter+=1
             total_nonpoi+=1
         #combined poi and non-poi    
         if data_dict[person][feature_name]!='NaN':
             full_counter+=1
         total_count+=1
            
                
     percent_full=round((float(full_counter)/total_count),2)
     percent_poi=round((float(poi_counter)/total_poi),2)
     percent_nonpoi=round((float(nonpoi_counter)/total_nonpoi),2)
     
     return (percent_full,full_counter), (percent_nonpoi,nonpoi_counter), \
     (percent_poi,poi_counter)

(percent_full,full_counter), (percent_nonpoi,nonpoi_counter), \
(percent_poi,poi_counter) = percent_count_coverage(feature,data_dict)
           
 # Calculate data coverage for each feature
for feature in financial_features_list:
	(percent_full,full_counter), (percent_nonpoi,nonpoi_counter), \
 (percent_poi,poi_counter) =  percent_count_coverage(feature,data_dict)
 
	print feature, "-", \
     (percent_full,full_counter), "-", (percent_nonpoi,nonpoi_counter), "-",\
     (percent_poi,poi_counter)
      
for feature in email_features_list:
	(percent_full,full_counter), (percent_nonpoi,nonpoi_counter), \
 (percent_poi,poi_counter) =  percent_count_coverage(feature,data_dict)
 
	print feature, "-", \
     (percent_full,full_counter), "-", (percent_nonpoi,nonpoi_counter), "-",\
     (percent_poi,poi_counter)       

#----------------------------------------------------
#---------------------------------------------------- 
        

### Task 3: Create new feature(s) and scale others
### Store to my_dataset for easy export below.

#Create features to look at fraction of emails to/from a POI or 

for name in data_dict.keys():
    from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
    to_messages = data_dict[name]["to_messages"]
    if (from_poi_to_this_person!='NaN') and (to_messages!='NaN'):   
        fraction_from_poi = float(from_poi_to_this_person)/float(to_messages)
        data_dict[name]["fraction_from_poi"] = fraction_from_poi
    else:
        data_dict[name]["fraction_from_poi"] = 0
        
    from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
    from_messages = data_dict[name]["from_messages"]
    if (from_this_person_to_poi!='NaN') and (from_messages!='NaN'):   
        fraction_to_poi = float(from_poi_to_this_person)/float(from_messages)
        data_dict[name]["fraction_to_poi"] = fraction_to_poi
    else:
        data_dict[name]["fraction_to_poi"] = 0

#Update Feature List 

email_features_list = [
    # 'email_address', # ignore email address; informational label
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    'fraction_from_poi', #new feature
    'fraction_to_poi',  #new feature
    ]
    
features_list = target_label + financial_features_list + email_features_list

#make a copy of the full list for use with testing
features_list_full=copy(features_list)

### Apply Min-Max feature scaling to all features
from sklearn import preprocessing
import math
import numpy as np

for feature in features_list:
	feature_values = []
	if feature == 'poi':
		continue
	else:	
		for name in data_dict:
			if math.isnan(float(data_dict[name][feature])) == False:
				### Extract feature values for scaling
				feature_values.append(float(data_dict[name][feature]))
			else:
				### Set 'NaN' values to 0 for use in sklearn
				feature_values.append(float(0))
				data_dict[name][feature] = float(0)
    
		### Fit scaler
		feature_values_array=np.array(feature_values)
		feature_values_array_fa=feature_values_array[:,np.newaxis]		
		min_max_scaler = preprocessing.MinMaxScaler()
		min_max_scaler.fit(feature_values_array_fa)

		for name in data_dict:
			### Apply scaler to values
   			data_dict[name][feature] = \
      min_max_scaler.transform([[float(data_dict[name][feature])]])[0,0]


### Store to my_dataset for easy export below.
my_dataset = copy(data_dict)

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#----------------------------------------------------
#----------------------------------------------------

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#Determine K-best features
#from sklearn.feature_selection import SelectKBest, f_classif
#
num_features = 22 # use all
#
#anova_filter = SelectKBest() #f_classif,k=num_features
#anova_filter.fit(features, labels)
#scores = anova_filter.scores_
#unsorted_pairs = zip(features_list[1:], scores)
#sorted_dict = sorted(unsorted_pairs, key=lambda feature: feature[1], reverse = True)
#anova_best_features = sorted_dict[:num_features]
#
#print "---Best 21 Features from Select KBest---"
#for item in anova_best_features:
#    print item[0],item[1]
    
#----------------------------------------------------
    
## Determine DecisionTree Feature Importances
from sklearn.tree import DecisionTreeClassifier

tree_filter = DecisionTreeClassifier()
tree_filter.fit(features, labels)
unsorted_pairs = zip(features_list[1:],tree_filter.feature_importances_)
sorted_dict = sorted(unsorted_pairs, key=lambda feature: feature[1], reverse = True)
tree_best_features = sorted_dict[:num_features]

print "---Sorted Features from Decision Tree Feature Importances---"
for item in tree_best_features:
    print item[0], item[1]

#----------------------------------------------------
#----------------------------------------------------

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Trim the features list to only look at the top 10 from SelectKBest

#feature_list= [
#      'poi',
#	'total_stock_value', 
#      'exercised_stock_options'
#	'bonus',
#	'salary', 
#	'deferred_income',
#	'long_term_incentive', 
#	'total_payments', 
#      'restricted_stock',
#	'shared_receipt_with_poi',  
#	'loan_advances'
#	]
 #Trim the features list to optimize f1-score using decision-tree ML algorithm
features_list= [
         'poi',
         'exercised_stock_options',
         'expenses',
         'from_this_person_to_poi'
         #'other',
         #'total_payments',
         #'bonus',
         #'shared_receipt_with_poi',
         #'from_poi_to_this_person',
         #'deferred_income'
         ]
                 
         
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

 ## Create Cross Validation object for use in GridSearchCV
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#from sklearn.pipeline import Pipeline
#----------------------------------------------------
#----------------------------------------------------

#Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#
## Build Pipeline
#clf = Pipeline([
# 	('anova', anova_filter),
#	 ('clf', GaussianNB())
#	 ])
##Algorithm determines optimal number of featurs  
#parameters = {
#     'anova__k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]			
# }
#
### Apply GridSearchCV to the dataset
#cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
#clf = GridSearchCV(clf, parameters, scoring='f1', cv=cv)
#clf.fit(features, labels)
#
#
#print("--Gaussian--")
#print(clf.best_estimator_)
#
##Validate and Evaluate
#clf = clf.best_estimator_
#test_classifier(clf, my_dataset, features_list_full)


#----------------------------------------------------
#----------------------------------------------------

# Decision Tree using most import DecisionTreeClassifer features
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier() 

parameters = {'min_samples_split': [2, 5, 10, 20, 30, 50],
              'criterion': ['gini', 'entropy']	
             }
             
#cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)                   
clf = GridSearchCV(clf, parameters, scoring='f1',cv=cv)
clf.fit(features, labels)

print("--Decision Tree--")
print(clf.best_estimator_)

#Validate and Evaluate (The top 3 features produce the highest f1 score)
clf = clf.best_estimator_
test_classifier(clf, my_dataset, features_list)


#----------------------------------------------------
#----------------------------------------------------
# AdaBoost 
#from sklearn.ensemble import AdaBoostClassifier
#
## Build Pipeline
#clf = Pipeline([
# 	('anova', anova_filter),
#	 ('clf', AdaBoostClassifier())
#	 ])
#
##Algorithm determines optimal number of features  
#parameters = {
#     'anova__k': [5,8,10,12,14,16,18,19,20,21],
#     'clf__n_estimators': [50,100,200],				
# }
#
### Apply GridSearchCV to the dataset
#
#cv = StratifiedShuffleSplit(labels, 500, random_state = 42) #use 500 iterations
#clf = GridSearchCV(clf, parameters, scoring='f1', cv=cv)
#clf.fit(features, labels)
#
#print("--Adaboost--")
#print(clf.best_estimator_)
#
##Validate and Evaluate
#clf = clf.best_estimator_
#test_classifier(clf, my_dataset, features_list_full)

#----------------------------------------------------
#----------------------------------------------------

##SVC
#from sklearn.svm import SVC
#
## Build Pipeline
#clf = Pipeline([
# 	('anova', anova_filter),
#	 ('clf', SVC())
#	 ])
#
#parameters = {
#     'anova__k': [5,8,10,12,14,16,18,19,20,21],
#     'clf__C': [1, 10, 100],  
#     'clf__kernel': ['linear','rbf']				
#     }
#
### Apply GridSearchCV to the dataset
#cv = StratifiedShuffleSplit(labels, 1000, random_state = 42) #use 10 iterations
#clf = GridSearchCV(clf, parameters, scoring='f1', cv=cv)
#clf.fit(features, labels)
#
#print("--SVC--")
#print(clf.best_estimator_)
#
##Validate and Evaluate
#clf = clf.best_estimator_
#test_classifier(clf, my_dataset, features_list_full)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)