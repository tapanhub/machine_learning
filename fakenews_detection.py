#!/usr/bin/env python
# coding: utf-8

# In[5]:


#ref: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news
#ref: https://github.com/GeorgeMcIntire/
#ref: https://github.com/GeorgeMcIntire/lingcon_workshop
#https://github.com/GeorgeMcIntire/lingcon_workshop/blob/master/Fake%20News%20Classifier%20Notebook%20Completed.ipynb
#https://www.kaggle.com/mrisdal/fake-news

'''
What is a TfidfVectorizer?
TF (Term Frequency): 
The number of times a word appears in a document is its Term Frequency. 
A higher value means a term appears more often than others, and so, the document is a good 
match when the term is part of the search terms.

IDF (Inverse Document Frequency): 
Words that occur many times a document, but also occur many times in many others, may be irrelevant. 
IDF is a measure of how significant a term is in the entire corpus.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

What is a PassiveAggressiveClassifier?
Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct 
classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. 
Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss,
causing very little change in the norm of the weight vector.
'''

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('news.csv')

print(df.shape)
df.head()
labels = df.label

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

