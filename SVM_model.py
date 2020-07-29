#This code is for SVM classification on textual features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import string
import nltk
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

df = pd.read_csv("Tanisha_dataset.csv",encoding='latin1')
df.head()

#Textual features

text_df = df[["title", "company_profile", "description", "requirements", "benefits","fraudulent"]]
text_df = text_df.fillna(' ')

text_df.head()

#Catagorical Feature
cat_df = df[["telecommuting", "has_company_logo", "has_questions", "employment_type", "required_experience", "required_education", "industry", "function","fraudulent"]]
cat_df = cat_df.fillna("None")

cat_df.head()

#Plotting the character count
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
length=text_df[text_df["fraudulent"]==1]['description'].str.len()
ax1.hist(length,bins = 20,color='orangered')
ax1.set_title('Fake Post')
length=text_df[text_df["fraudulent"]==0]['description'].str.len()
ax2.hist(length, bins = 20)
ax2.set_title('Real Post')
fig.suptitle('Characters in description')
plt.show()

#Concate the text data for preprocessing and modeling
text = text_df[text_df.columns[0:-1]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
target = df['fraudulent']

#Tokenizing text
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
text = text.apply(lambda x: tokenizer.tokenize(x))
text.head(3)

text = text.apply(lambda x : combine_text(x))
text.head(3)

x=text
y=target
msg_train,msg_test,class_train,class_test=train_test_split(x,y, test_size=0.5, random_state=42)


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(msg_train.values)

#Building model
classifier= svm.SVC(gamma='scale',kernel='poly',degree=2,random_state=1)
targets=class_train.values
classifier.fit(counts, targets.ravel())

test_count=vectorizer.transform(msg_test)
predictions=classifier.predict(test_count)

#confusion matrix
CM = confusion_matrix(class_test,predictions)

TN=CM[1][1]
FN=CM[0][1]
TP=CM[0][0]
FP=CM[1][0]

print("True Negative=",TN)
print("False Negative=",FN)
print("True Positive=",TP)
print("False Positive=",FP)
        
recall=CM[1][1]/(CM[1][1]+CM[1][0])
print("recall=",recall)

precision=CM[1][1]/(CM[1][1]+CM[0][1])
print("precision=",precision)

f1_score=2*((precision*recall)/(precision+recall))
print("f1 score=",f1_score)

accuracy=(CM[0][0]+CM[1][1])/(CM[0][0]+CM[1][1]+CM[0][1]+CM[1][0])
print("accuracy=",accuracy)
