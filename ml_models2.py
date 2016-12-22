
# coding: utf-8

# In[1]:

import pickle
data=pickle.load( open( "data_with_text", "rb" ) )
data.head()


# In[2]:

data.shape


# In[3]:

import pandas as pd
rt_set = pd.read_csv('SP500_data.csv')


# In[4]:

data['return'] = rt_set['adj.close'].values[:-1]
data.head()


# In[5]:

ten_year = [0, 2511, 5000, 7526, 10054, 12582, 15097, data.shape[0]]


# In[6]:

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB


# In[7]:

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LogisticRegression()),])
acc = []
auc = []
for i in range(len(ten_year)-1):
    print ten_year[i],ten_year[i+1]
    datause = data.loc[ten_year[i]:ten_year[i+1],['ret','sentence']]
    X_train, X_test, y_train, y_test = train_test_split(datause['sentence'].values, datause['ret'].values, test_size=0.2, random_state=42)
    
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    acc.append(np.mean(predicted == y_test))
    auc.append(roc_auc_score(y_test, predicted))
    print acc[i], auc[i]


# In[8]:

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', linear_model.SGDClassifier()),])
acc2 = []
auc2 = []
for i in range(len(ten_year)-1):
    print ten_year[i],ten_year[i+1]
    datause = data.loc[ten_year[i]:ten_year[i+1],['ret','sentence']]
    X_train, X_test, y_train, y_test = train_test_split(datause['sentence'].values, datause['ret'].values, test_size=0.2, random_state=42)
    
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    acc2.append(np.mean(predicted == y_test))
    auc2.append(roc_auc_score(y_test, predicted))
    print acc2[i], auc2[i]


# In[9]:

acc3 = []
auc3 = []
for i in range(len(ten_year)-1):
    print ten_year[i],ten_year[i+1]
    datause = data.loc[ten_year[i]:ten_year[i+1],['ret','sentence','return']]
    
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform(datause['sentence'].values)
    features = tfidf.toarray()
    X_train, X_test, y_train, y_test = train_test_split(features, datause['ret'].values, test_size=0.2, random_state=42)
    
    clf = GaussianNB()
    text_clf = clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    acc3.append(np.mean(predicted == y_test))
    auc3.append(roc_auc_score(y_test, predicted))
    print acc3[i], auc3[i]


# In[10]:

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier()),])
acc4 = []
auc4 = []
for i in range(len(ten_year)-1):
    print ten_year[i],ten_year[i+1]
    datause = data.loc[ten_year[i]:ten_year[i+1],['ret','sentence']]
    X_train, X_test, y_train, y_test = train_test_split(datause['sentence'].values, datause['ret'].values, test_size=0.2, random_state=42)
    
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    acc4.append(np.mean(predicted == y_test))
    auc4.append(roc_auc_score(y_test, predicted))
    print acc4[i], auc4[i]


# In[17]:

import matplotlib.pyplot as plt

def auc_acc_plot(acc,auc,name,random):
    plt.figure(random)
    plt.plot(acc,label='Accuracy')
    plt.plot(auc,label='ROC_AUC')
    plt.title('Model Accuracy for Every Ten Year - '+name)
    plt.xlabel('Years')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(7),['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s'])
    plt.savefig(name+'.png')


# In[18]:

auc_acc_plot(acc,auc,'LR',123)
auc_acc_plot(acc2,auc2,'SGD',234)
auc_acc_plot(acc3,auc3,'NB',345)
auc_acc_plot(acc4,auc4,'RF',456)


# In[19]:

acc5 = []
auc5 = []
for i in range(len(ten_year)-1):
    print ten_year[i],ten_year[i+1]
    datause = data.loc[ten_year[i]:ten_year[i+1],['ret','sentence','return']]
    
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform(datause['sentence'].values)
    features = tfidf.toarray()
    X_train, X_test, y_train, y_test = train_test_split(features, datause['ret'].values, test_size=0.2, random_state=42)
    
    rt = [[t] for t in datause['return'].values]
    
    features = np.hstack((features,rt))
    
    clf = LogisticRegression()
    text_clf = clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    acc5.append(np.mean(predicted == y_test))
    auc5.append(roc_auc_score(y_test, predicted))
    print acc5[i], auc5[i]


# In[20]:

def auc_acc_plot2(acc,auc,name,random):
    plt.figure(random)
    plt.plot(acc,label='Accuracy')
    plt.plot(auc,label='ROC_AUC')
    plt.plot(acc,label='Accuracy_tfidf+return')
    plt.plot(auc,label='ROC_AUC_tfidf+return')
    plt.title('Model Accuracy for Every Ten Year - '+name)
    plt.xlabel('Years')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(7),['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s'])
    plt.savefig(name+'.png')


# In[21]:

auc_acc_plot2(acc5,auc5,'LR',1235)


# In[ ]:



