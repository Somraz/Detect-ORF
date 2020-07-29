import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from nltk.corpus import stopwords
import nltk

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

pd.set_option('display.max_columns', None)

#Reading dataset and dropping irrlevant columns
df = pd.read_csv("Tanisha_dataset.csv",encoding='latin1')
df2 = df.copy()
df2.drop(['location', 'salary_range', 'department'], axis = 1, inplace = True)
df2 = df2.fillna('NULL')

#Combining all the text features
df2['description'] = df2['description'] + ' ' + df2['requirements'] + ' ' + df2['company_profile']
df2.drop(['company_profile', 'requirements', 'benefits'], axis = 1, inplace = True)

#Tokenizing the text
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
df2['description'] = df2['description'].apply(lambda x: tokenizer.tokenize(x))
df2['description'] = df2['description'].apply(lambda x : combine_text(x))

#Converting text to bag of words model using CountVectorizer
vectorizer = CountVectorizer(ngram_range = (1,3), min_df = 0.06)
vectorizer_features = vectorizer.fit_transform(df2['description'])

vectorized_df = pd.DataFrame(vectorizer_features.todense(), columns = vectorizer.get_feature_names())
vectorized_df.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df_final = pd.concat([df2, vectorized_df], axis = 1)

df_final.drop('description', axis = 1, inplace = True)
df_final.dropna(inplace=True)

#One hot encoding of categorical features
columns_to_1_hot = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for column in columns_to_1_hot:
    encoded = pd.get_dummies(df_final[column])
    df_final = pd.concat([df_final, encoded], axis = 1)
columns_to_1_hot += ['title']
df_final.drop(columns_to_1_hot, axis = 1, inplace = True)


#Train test split of data
target = df_final['fraudulent']
features = df_final.drop('fraudulent', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, stratify = target, random_state=42)

#Building model
svc = SVC()
kernel = ['linear', 'rbf']
param_grid_knn = dict(kernel = kernel)
print (param_grid_knn)
grid_svc = GridSearchCV(svc, param_grid_knn, cv = 10, scoring = 'roc_auc', n_jobs = -1, verbose = 2)
grid_svc.fit(X_train, y_train)
svc_pred = grid_svc.predict(X_test)

#Result metrics
print (roc_auc_score(y_test, svc_pred))
print (classification_report(y_test, svc_pred))
print (confusion_matrix(y_test, svc_pred))
