import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter

pd.set_option('display.max_columns', None)

#Reading the csv file
df = pd.read_csv("Tanisha_dataset.csv",encoding='latin1')
df2 = df.copy()
df2.drop(['location', 'salary_range', 'department'], axis = 1, inplace = True)
df2 = df2.fillna('NULL')

#Combining all the textual features
df2['description'] = df2['description'] + ' ' + df2['requirements'] + ' ' + df2['company_profile'] + ' ' + df2['benefits']
df2.drop(['company_profile', 'requirements','benefits'], axis = 1, inplace = True)
print(df2.head())


#Most frequent words plotting for fake jobs

Fraud_1 = [text for text in df2[df2['fraudulent'] == 1]['description']]
Fraud_1 = ' '.join(Fraud_1).split()
Fraud_1_counts = Counter(Fraud_1)
Fraud_1_common_words = [word[0] for word in Fraud_1_counts.most_common(20)]
Fraud_1_common_counts = [word[1] for word in Fraud_1_counts.most_common(20)]
fig = plt.figure(figsize = (20, 10))
pal = sns.color_palette("cubehelix", 20)
sns.barplot(x = Fraud_1_common_words, y = Fraud_1_common_counts, palette=pal)
plt.title('Most Common Words used in Fake job postings')
plt.ylabel("Frequency of words")
plt.xlabel("Words")
plt.show()

#Most frequent words plotting for legitimate jobs

Fraud_0 = [text for text in df2[df2['fraudulent'] == 0]['description']]
Fraud_0 = ' '.join(Fraud_0).split()
Fraud_0_counts = Counter(Fraud_0)
Fraud_0_common_words = [word[0] for word in Fraud_0_counts.most_common(20)]
Fraud_0_common_counts = [word[1] for word in Fraud_0_counts.most_common(20)]
fig = plt.figure(figsize = (20, 10))
pal = sns.color_palette("cubehelix", 20)
sns.barplot(x = Fraud_0_common_words, y = Fraud_0_common_counts, palette=pal)

plt.title('Most Common Words used in Genuine job postings')
plt.ylabel("Frequency of words")
plt.xlabel("Words")
plt.show()
