#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing modules
import pandas as pd
import numpy as np
import os
import nltk
import string
import re

from  nltk.corpus import stopwords

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_style('whitegrid')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('averaged_perceptron_tagger')

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS 
from nltk import word_tokenize, pos_tag

from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
import webbrowser

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# In[10]:


def text_process(mess):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~' 

#     nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=[char for char in mess if char not in punctuations]
    nopunc=''.join(nopunc)
    
    nostopwords= [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(nostopwords)


# In[11]:


def nouns_adj(text,flag):
#     is_noun_adj=lambda pos:pos[:2]=='NN'  #or pos[:2]=='JJ'
    is_noun_adj=lambda pos:pos[:2]==flag  #or pos[:2]=='JJ'
    tokenised=word_tokenize(text)
    all_nouns_adj=[word for (word, pos) in nltk.pos_tag(tokenised) if is_noun_adj(pos)]
    all_nouns_adj=  ' '.join(all_nouns_adj)
    return all_nouns_adj


# In[17]:


def eivatable1():
    chatlog = pd.read_csv(r'C:\Users\612890877\Desktop\Wordmeter\conversationlog_Train.csv', encoding = "ISO-8859-1", low_memory=False)

    # Print head
    #print(chatlog.head())

    duns = pd.read_csv(r'C:\Users\612890877\Desktop\Wordmeter\Dunnes.csv')
    # Filter EIVA
    chatlog=chatlog[chatlog['agentid']=='Eiva']

    # chatlog=chatlog[chatlog['intentname']!='greetings']
    chatlog.head()

    #useful columns - 'conversationreq', 'intentname', 'product', 'dunsid' - extract is for 13th oct
    chatlog = chatlog[['conversationreq','conversationid','conversationmessageid', 'intentname', 'product', 'dunsid']]
    # chatlog.head()
    chatlog.info()

    #Intent cleanse as per EIVA rules
    arrApp= ['book_appointment','confirm_appointment','request_appointment_slots',
             'retrieve_appointment','req_updates_and_appointment','reject_appointment',
             'update_appointment','appointment']

    arrOrder=['order_updates','request_new_order_ref','pre_order_ref_confirm','order_ref_input','track_fault',
    'fault','milestone','order','engineering_notes','confirmation','linked_order','get_kci_status','product_not_supported',
    'fdt_error_code_002','fdt_error_code_003','simtwo_details','lorn_details','unauth_access_001','unauthorized_dunsid',
    'reference_num_query','fdt_error_code_001','fdt_error_code_004','fdt_error_code_005','fdt_returned_error']

    arrchitchat=['chitchat','greetings','default_response','thanks','thanks_positive','restart_flow','start','done',
                 'welcome_openreach','end_conversation','further_assistance']

    arrconnect=['connect_to_live_agent','talk_to_liveagent','connect_to_agent_using_service_id']

    arrGetsitecontact=['get_sitecontact_details','site_contact_update_not_allowed','update_sitecontact_details']

    arrOthers =['send_kci','resend_kci','Blanks','UNKNOWN','completion_date','training','other','ambigious',
                'intent_not_clear','form_intent','intent_dismiss']

    # chatlog['sentimenttext']=chatlog['sentiment'].apply(lambda x: 'Positive' if x>0.05 else('Negative' if x<-0.05 else 'Neutral')  )
    # chatlog['EIVAIntent'] =chatlog['intentname'].apply(lambda x: 'Appointment' if x in ['book_appointment','confirm_appointment'])
    chatlog['EIVAIntent'] =chatlog['intentname'].apply(lambda x: 'Appointment' if any([k in x for k in arrApp]) else
                                                      ('TrackOrder' if any([k in x for k in arrOrder]) else
                                                      ('Chitchat' if any([k in x for k in arrchitchat]) else 
                                                       ('Connect_to_LIVE_Agent' if any([k in x for k in arrconnect]) else 
                                                        ('GetSiteContact' if any([k in x for k in arrGetsitecontact]) else
                                                         ('Others' if any([k in x for k in arrOthers]) else x
                                                         )
                                                        )
                                                      )
                                                      )
                                                      )
                                                      )

    #1
    chatlog['product'].fillna('None',inplace=True)

    # 2
    chatlog['conversationreq'].fillna('',inplace=True)

    # 3 convert numeric only cells with na eg: '324242423' as ''
    chatlog['conversationreq_processed'] = chatlog['conversationreq'] .apply(lambda item : '' if str(item).isdigit() else item)

    #4 process converse_req to remove punctuation and stop words
    chatlog['conversationreq']=chatlog['conversationreq_processed'].apply(str) 
    chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(text_process)
    chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(str) 

    #5 Remove numbers from a string eg: 'Track order 123123' as 'Track order'
    chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(lambda x: re.sub('\d', '', str(x)))


    #6 Connect Desk Agent, Track Order - join with '_' in  conversationreq_processed
    chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(lambda x: x.strip().replace(r" ","_") if x.strip().lower()=='connect desk agent' else x)
    chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(lambda x: x.strip().replace(r" ","_") if x.strip().lower()=='track order' else x)


    #7 len of original request string
    chatlog['conversationreq_len']=chatlog['conversationreq'].apply(str).apply(len)
    #chatlog.conversationreq_processed.head(50)

    p='(OR011-\d+|OR01\d+|OR02\d+|OGEA\d+|OS01\d+|OS02\d+|OFVA\d+|SGEA\d+|LLSS\d+|STSS\d+|LMSS\d+|0141\d+|LLCM\d+|1-\d+|2-\d+|5-7-\d+|5-8-\d+|A\d+|A0\d+|A1\d+|B0\d+|S0\d+|A5\d+|A6\d+|A9\d+|B1\d+|S1\d+|A2\d+|ONP\d+|PRH\d+|UUB\d+|NUN\d+|QBJ\d+|LLS\d+|FBC\d+|LLL\d+|LUE\d+|PAR\d+|UFB\d+|MFB\d+|FBM\d+|TEZ\d+|FBW\d+|KBB\d+|OOF\d+|NEN\d+|FBL\d+|LLEA\d+|LLEM\d+|EVZ\d+|LYE\d+|UWR\d+|UNB\d+|FBN\d+|FBS\d+|LYW\d+|FBT\d+|OOZ\d+|MTU\d+|OGE\d+|TBP\d+|UWB\d+|LNC\d+|LTL\d+|OFP\d+|UBB\d+|OZP\d+|LKC\d+|QRD\d+|PWH\d+|OVP\d+|LLWS\d+|LLWW\d+|LLWN\d+|LLLW\d+|LLWM\d+|LLWE\d+|LLWR\d+|OVZ\d+|OHP\d+|NNX\d+|TUZ\d+|FBE\d+|LLM\d+|LWE\d+|LTE\d+|LLN\d+|OZZ\d+|PPR\d+|QSJ\d+)'

    chatlog['valid_order']=chatlog['conversationreq'].str.extract(p, expand=True)

    # #convert dunsid to string
    duns['dunsid'] = duns['dunsid'].fillna(0).apply(int)
    duns['dunsid'] = duns['dunsid'].apply(str)
    duns=duns[duns['sm_disabled_flag']=='Active']
    #duns.head()

    dunsunique=duns[['dunsid','cp_name']].drop_duplicates()

    # #join chat and duns to get CP name
    chatlog =pd.merge(chatlog,dunsunique ,how='left',on='dunsid')
    # chatlog.head()
    #chatlog.info()
    dfintent = chatlog[['conversationid','conversationmessageid','conversationreq_processed']]

    dfintent['Nouns']=dfintent['conversationreq_processed'].apply(lambda x: nouns_adj(text=x,flag='NN'))

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(dfintent['Nouns'])  #try conversationreq_processed

    #plot_10_most_common_words(count_data, count_vectorizer,dftable)

    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:20] #### top 20 words

    words = [w[0] for w in count_dict]
    counts = [int(w[1]) for w in count_dict]

    retdf=pd.DataFrame({'Aspect':np.array(words),'Count':np.array(counts)})
    retdf=retdf.to_dict('records')
    return retdf






