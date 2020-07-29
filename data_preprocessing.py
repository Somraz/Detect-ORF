#This code was used to pre-process textual features

import re
import csv
import os
import pandas as pd
from string import punctuation
from nltk import stem
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ftfy import fix_text

stopwords = set(stopwords.words('english'))
stemmer = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def lemmetization(text):
    
    text = " ".join([stemmer.stem(word) for word in text.split()])
    text = " ".join([lemmatizer.lemmatize(word, pos='v') for word in text.split()])
    return text

def stop_words(text):

    text=fix_text(text)
    text=fix_encoding(text)
    return text

def url1(text):
    text = " ".join([word for word in text.split() if not word.startswith('url')])
    return text
    

def clean_email(text):
    
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub("\d+", " ", text)
    text = text.replace('\n', ' ')
    text = text.lower()
    text = re.sub('\W+',' ',text )
    text = re.sub(r"[,.;@#?!&$]+\ *", " ", text)
    return text

#All the changes were made in this loop to perform preprocessing
with open('Tanisha_dataset.csv', 'r',encoding="latin1") as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)
for data in lines:
    if data[9]=='f':
       data[9]=0;
    else:
        data[9]=1

                
#Writing back all the changes on the file
with open('dataset1.csv', 'w', newline='',encoding="latin1") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
readFile.close()
writeFile.close()
     




