import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from nltk.corpus import stopwords
import nltk
from sklearn.svm import OneClassSVM 


def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

pd.set_option('display.max_columns', None)

df = pd.read_csv("Tanisha_dataset.csv",encoding='latin1')
df2 = df.copy()
df2.drop(['location', 'salary_range', 'department'], axis = 1, inplace = True)
df2 = df2.fillna('NULL')

#Dropping the irrelevant columns
df2.drop(['description','company_profile', 'requirements', 'benefits'], axis = 1, inplace = True)

#One-hot encoding of features.
columns_to_1_hot = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for column in columns_to_1_hot:
    encoded = pd.get_dummies(df2[column])
    df2= pd.concat([df2, encoded], axis = 1)
columns_to_1_hot += ['title']
df2.drop(columns_to_1_hot, axis = 1, inplace = True)


#Performing train test split
target = df2['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(df2, target, test_size = 0.2, stratify = target, random_state=42)

#Number of normal data points in training set
X_train_normal = X_train[X_train['fraudulent']==0]

#Number of outliers in training set
X_train_outliers = X_train[X_train['fraudulent']==1]

#Building model
outlier_prop = len(X_train_outliers) / len(X_train_normal) 
svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.3) 
svm.fit(X_train_normal.drop('fraudulent', axis = 1))

X_test=X_test.drop('fraudulent', axis = 1)
svc_pred = svm.predict(X_test)
svc_pred = np.array([y==-1 for y in svc_pred])

#Result metrics
print (roc_auc_score(y_test, svc_pred))
print (classification_report(y_test, svc_pred))
print (confusion_matrix(y_test, svc_pred))




