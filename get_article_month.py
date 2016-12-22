
# coding: utf-8

# In[1]:

from nytimesarticle import articleAPI
import pandas as pd
import datetime
import time
import os
import urllib2
import pandas as pd
import simplejson as json
from itertools import chain
#api = archiveAPI('4dd9ed2dfe144c63a2a1f852959a8fd1')


# In[2]:

sp = pd.read_csv('SP500_data.csv')
sp['date'].values


# In[4]:

#59683c2406c5440daf482674b11af021
#4dd9ed2dfe144c63a2a1f852959a8fd1
nyt_key='59683c2406c5440daf482674b11af021'
nyt_api='http://api.nytimes.com/svc/archive/v1/'


# In[5]:

import sys
reload(sys)
sys.setdefaultencoding('utf8')


# In[6]:

def if_have_key(d):
    try: 
        return d.get('main') 
    except:
        return d


# In[7]:

for year in range(2016,2017):
    for month in range(12,13):
        dicts = []
        url = nyt_api + str(year) + '/' + str(month) + '.json?api-key=' + nyt_key
        print url
        response = urllib2.urlopen(url).read()
        dict_response = json.loads(response, encoding = 'utf-8')
        response = dict_response['response']
        docs = response['docs']
        dicts.append(docs)
        data = pd.DataFrame(list(chain.from_iterable(dicts)))
        data['pub_date'] = [d.split("T")[0] for d in data['pub_date'].values]
        
        a=data[['headline','lead_paragraph','pub_date','web_url','word_count']]
        a.headline = [if_have_key(d) for d in a.headline]
        
        a.to_csv('./month/'+str(year)+str(month)+'_test.csv', index=False)
        time.sleep(2)
