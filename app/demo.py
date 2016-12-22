
# coding: utf-8

# In[1]:

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle



def combine(csv, text, model):
    data = pd.read_csv(str(csv))
    text=pickle.load( open( str(text), "rb" ) )
    data['sentence'] = text.tail()['sentence'].values
    
    model=pickle.load( open(str(model), "rb" ) )
    
    predicted = model.predict(data['sentence'].values)
    
    result = ['Up' if item == 1 else 'Down' for item in predicted]
    print "Results:",result
    real = ['Up' if item == 1 else 'Down' for item in data['ret'].values]
    print "Real:",real


# In[5]:

import sys
csv = "demo_returns.csv"
text = "data_with_text"
model = "model"
combine(csv, text, model)


# In[ ]:



