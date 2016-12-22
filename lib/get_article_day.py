
# coding: utf-8

# In[1]:

from nytimesarticle import articleAPI
import pandas as pd
import datetime
import time
api = articleAPI('c1662964fbdb4869a6d2f52d1e2dfd01')
api2 = articleAPI('59683c2406c5440daf482674b11af021')
api3 = articleAPI('77f615a24c244223a64cc7a082bb22a7')
api4 = articleAPI('6cd7ad57fe1649d09eaaf6e07f7cbd0d')
api5 = articleAPI('289bb634ebb34d6baf7f78e765957dac')


# In[2]:

sp = pd.read_csv('SP500_data.csv')
sp['date'].values


# In[16]:

count = 198
for date in sp['date'].values[198:900]:
    begin_date = end_date = datetime.datetime.strptime(date, '%m/%d/%Y').strftime('%Y%m%d')
    print count, begin_date
    headline = []
    lead_p = []
    
    for page in range(10):
        time.sleep(2)
        try:
            articles = api3.search(fl=['headline','lead_paragraph'], begin_date = int(begin_date),end_date=int(begin_date), page = page)
        
        #articles = api2.search(fl=['headline','lead_paragraph'], begin_date = int(begin_date),end_date=int(begin_date), page = page)
             
        #articles = api3.search(fl=['headline','lead_paragraph'], begin_date = int(begin_date),end_date=int(begin_date), page = page)
                
        #articles = api4.search(fl=['headline','lead_paragraph'], begin_date = int(begin_date),end_date=int(begin_date), page = page)
        
        #articles = api5.search(fl=['headline','lead_paragraph'], begin_date = int(begin_date),end_date=int(begin_date), page = page)
            a = articles['response']['docs']
            for x in a:
                headline.append(x['headline']['main'])
                lead_p.append(x['lead_paragraph'])
        
        except ValueError:
            pass
        
    thefile = open('./headline/'+begin_date+'_h.txt', 'w')
    for item in headline:
        thefile.write("%s\n" % item)
    thefile.close() 
    thefile = open('./lead_pragraph/'+begin_date+'_p.txt', 'w')
    for item in lead_p:
        thefile.write("%s\n" % item)
    thefile.close()
    count+=1

