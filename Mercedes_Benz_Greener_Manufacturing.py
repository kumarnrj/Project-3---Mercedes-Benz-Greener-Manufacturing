# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:54:46 2019

@author: NR
"""
'''
# Following actions should be performed:
* If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
* Check for null and unique values for test and train sets
* Apply label encoder.
* Perform dimensionality reduction.
* Predict your test_df values using xgboost
'''

# Importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Checking the Null value
train.isnull().sum()
test.isnull().sum()

# cheking the unique values in the dataset

list ={}
for col in train.columns:
    print(train[col].unique()," ",col)
    un = str(np.sort(train[col].unique()).tolist())
    tlist = list.get(un,[])
    tlist.append(col)
    list[un] = tlist[:]

for unique_val, column in list.items():
    print("unique value ",unique_val)
    print(column)
    print("\n")
    
# The columns whicha are having unique value is equal to 0 their variance will also be zero
# ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']
 
# checking the variance for above variable
np.var(train['X11'])

# cheking the unique values in the dataset of test

list1 ={}
for col in test.columns:
    print(train[col].unique()," ",col)
    un1 = str(np.sort(test[col].unique()).tolist())
    tlist1 = list1.get(un1,[])
    tlist1.append(col)
    list1[un1] = tlist1[:]

for unique_val, column in list1.items():
    print("unique value ",unique_val)
    print(column)
    print("\n")

# The columns whicha are having unique value is equal to 0 their variance will also be zero
# ['X257', 'X258', 'X295', 'X296', 'X369']
# checking the variance for above variable
np.var(train['X257'])


# separating the train dataset
X_train =np.asarray( train.drop(columns=['ID','y','X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']))
y_train = train.iloc[:,1].values

X_test = np.asarray(test.drop(columns=['ID','X257', 'X258', 'X295', 'X296', 'X369']))

# Applying label encoder on trainnig dataset
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(0,9):
    X_train[:,i] = labelencoder.fit_transform(X_train[:,i])

# Applying label encoder on testing  dataset
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(0,9):
    X_test[:,i] = labelencoder.fit_transform(X_test[:,i])



# # dimension redution
from sklearn.decomposition import PCA

pca = PCA(n_components=2,random_state =0)  
X_train = pca.fit_transform(X_train) 

# # dimension redution
from sklearn.decomposition import PCA

pca = PCA(n_components=2,random_state =0)  
X_test = pca.fit_transform(X_test) 


# Fitting the model 
from xgboost import XGBRegressor
reg = XGBRegressor(objective='reg:linear',random_state=300)
reg.fit(X_train,y_train)

# Predicting the value
y_pred = reg.predict(X_test)

#chekcing the accurecy
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=reg,X=X_train,y=y_train,cv=10)
acc.mean()
acc.std()


# Applying grid search
from sklearn.model_selection import GridSearchCV
parameteres = [ { 'base_score': [0.5,1,1.5], 'booster': ['gbtree']},
                 {'base_score': [0.5,1,1.5], 'booster': ['gblinear'],'gamma':[0.5,0.1,0.01,0.001,0.0001],'reg_alpha':[0,1,2]},
                 {'base_score': [0.5,1,1.5], 'booster': ['dart'],'gamma':[0.5,0.1,0.01,0.001,0.0001],'reg_alpha':[0,1,2]}]
grid_search = GridSearchCV(estimator = reg,
                           param_grid = parameteres,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_





# Appending the trianing and testing dataset
dataset = train.append(test,sort =False)




train_y = np.asarray(dataset[dataset['y'].notnull()]['y'])
train_X = np.asarray(dataset[dataset['y'].notnull()].drop(columns = ['ID', 'y']))
test_y = np.asarray(dataset[dataset['y'].isnull()].drop(columns = ['ID','y']))

# Applying label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(0,9):
    train_X[:,i] = labelencoder.fit_transform(train_X[:,i])

# Applying label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(0,9):
    test_y[:,i] = labelencoder.fit_transform(test_y[:,i])
    
'''#dimension reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=300,random_state =0)  
train_X = pca.fit_transform(train_X) '''


# Fitting the model 
from xgboost import XGBRegressor
reg = XGBRegressor(base_score = 0.5,booster ='dart',gamma = 0.1,)
reg.fit(train_X,train_y)

y_pred = reg.predict(test_y)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=reg,X=train_X,y=train_y,cv=10)
acc.mean()
acc.std()

# Applying grid search
from sklearn.model_selection import GridSearchCV
parameteres = [ { 'base_score': [0.5,1,1.5], 'booster': ['gbtree']},
                 {'base_score': [0.5,1,1.5], 'booster': ['gblinear'],'gamma':[0.5,0.1,0.01,0.001,0.0001],'reg_alpha':[0,0.1,0.01,0.001]},
                 {'base_score': [0.5,1,1.5], 'booster': ['dart'],'gamma':[0.5,0.1,0.01,0.001,0.0001],'reg_alpha':[0,0.1,0.01,0.001]}]
grid_search = GridSearchCV(estimator = reg,
                           param_grid = parameteres,
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_