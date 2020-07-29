import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
import string
import re
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import nltk
from sklearn.utils import resample

pd.set_option('display.max_columns', None)

df = pd.read_csv("Tanisha_dataset.csv",encoding='latin1')

#Balancing the dataset
df_majority = df[df.fraudulent==0]
df_minority = df[df.fraudulent==1]
 
df_majority_downsampled = resample(df_majority, replace=False,n_samples=868, random_state=123) 
df = pd.concat([df_majority_downsampled, df_minority])

df2 = df.copy()
df2.drop(['location', 'salary_range', 'department'], axis = 1, inplace = True)
df2 = df2.fillna('NULL')

#Dropping all the irrelevant features
df2.drop(['description','company_profile', 'requirements', 'benefits'], axis = 1, inplace = True)

#One-hot encoding on categorical features.
columns_to_1_hot = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for column in columns_to_1_hot:
    encoded = pd.get_dummies(df2[column])
    df2= pd.concat([df2, encoded], axis = 1)
columns_to_1_hot += ['title']
df2.drop(columns_to_1_hot, axis = 1, inplace = True)
print(df2.head()) 

target = df2['fraudulent']
X_train=df2.copy() 

#Building mofel
km = KMeans(n_clusters=2)
km.fit(X_train.drop('fraudulent', axis = 1))
km_pred = km.predict(X_train.drop('fraudulent', axis = 1))

labels = km.labels_

#result metrics
print (roc_auc_score(df2['fraudulent'], labels))
print (classification_report(df2['fraudulent'], labels))
print (confusion_matrix(df2['fraudulent'], labels))

