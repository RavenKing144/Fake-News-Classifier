#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import nltk
import re
import numpy as np
import itertools
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier


# In[4]:


df = pd.read_csv("C:/Users/SarthakAgarwal/Downloads/NLP Projects/Fake news Classifier/dataset/dataset/train.csv")


# In[12]:


df = df.dropna()


# In[13]:


message = df.copy()


# In[14]:


message.reset_index(inplace=True)


# In[23]:


lematizer = WordNetLemmatizer()


# In[108]:


corpus = []
stopword = set(stopwords.words('english'))
for i in message.values:
    title = i[-2].lower()
    review = re.sub('[^a-z]', ' ', title).split()
    review = " ".join([lematizer.lemmatize(j) for j in review if j not in stopword])
    corpus.append(review)


# In[109]:


# cv = CountVectorizer(max_features=6000, ngram_range=(1,5))
# x = cv.fit_transform(corpus).toarray()
tf_vc = TfidfVectorizer(max_features=6000, ngram_range=(1,5))
x = tf_vc.fit_transform(corpus).toarray()


# In[110]:


y = message.label


# In[111]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[112]:


classifier = MultinomialNB()


# In[113]:


classifier.fit(x_train,y_train)


# In[114]:


pred = classifier.predict(x_test)


# In[115]:


score = metrics.accuracy_score(y_test, pred)
cm = metrics.confusion_matrix(y_test, pred)


# In[116]:


score


# In[117]:


cm


# In[118]:


linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(x_train, y_train)
pred = linear_clf.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
cm = metrics.confusion_matrix(y_test, pred)


# In[119]:


score


# In[120]:


cm


# In[121]:


linear_clf = MultinomialNB(alpha = 0.7)
linear_clf.fit(x_train, y_train)
pred = linear_clf.predict(x_test)
score = metrics.accuracy_score(y_test, pred)
cm = metrics.confusion_matrix(y_test, pred)


# In[122]:


score


# In[123]:


cm


# In[ ]:





# In[ ]:





# In[ ]:




