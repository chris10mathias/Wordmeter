# Importing modules
import pandas as pd
import numpy as np
import os
import nltk
import string
import re

from  nltk.corpus import stopwords

import matplotlib.pyplot as plt
#%matplotlib inline

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
os.chdir('..')
chatlog = pd.read_csv(r'C:\Users\612890877\Desktop\Wordmeter\conversationlog_Train_2 - mask.csv',encoding = "ISO-8859-1", low_memory=False)

# Print head
#print(chatlog.head())

duns = pd.read_csv(r'C:\Users\612890877\Desktop\Wordmeter\Dunnes.csv')
cnt=1

print("cp before",duns['cp_name'].nunique())

for i in duns['cp_name'].unique():

    new='customer_' + str(cnt)

    duns.loc[duns['cp_name'] == i, ['cp_name']] = new

#     print(new)

    cnt=cnt+1


def text_process(mess):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~' 

#     nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=[char for char in mess if char not in punctuations]
    nopunc=''.join(nopunc)
    
    nostopwords= [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(nostopwords)

def nouns_adj(text,flag):
#     is_noun_adj=lambda pos:pos[:2]=='NN'  #or pos[:2]=='JJ'
    is_noun_adj=lambda pos:pos[:2]==flag  #or pos[:2]=='JJ'
    tokenised=word_tokenize(text)
    all_nouns_adj=[word for (word, pos) in nltk.pos_tag(tokenised) if is_noun_adj(pos)]
    all_nouns_adj=  ' '.join(all_nouns_adj)
    return all_nouns_adj

def get_sentiment(r):
    sentiment_score = vader.polarity_scores(r)
    score = sentiment_score['compound']     
    return score 

# Filter EIVA
#chatlog=chatlog_[chatlog_['agentid']=='Eiva']
#chatlog['con_request_time'] = chatlog['con_request_time'].apply(lambda x: pd.to_datetime(str(x), format='%d/%m/%Y %H:%M:%S')) 
#chatlog['con_response_time'] = chatlog['con_response_time'].apply(lambda x: pd.to_datetime(str(x), format='%d/%m/%Y %H:%M:%S')) 
chatlog['con_request_time'] = pd.to_datetime(chatlog['con_request_time'], format='%d/%m/%Y %H:%M:%S')
chatlog['con_response_time'] = pd.to_datetime(chatlog['con_response_time'], format='%d/%m/%Y %H:%M:%S')

chatlog['Duration'] = (chatlog.con_response_time-chatlog.con_request_time)/np.timedelta64(1,'s')
chatlog['Duration'].fillna(0, inplace=True) 

chatlog['conversationid']=chatlog['conversationid'].astype(str)
chatlog['conversationmessageid']=chatlog['conversationmessageid'].astype(str)



#chatlog['conversationreq'].fillna('', inplace=True)
#chatlog['conversationres'].fillna('', inplace=True) 


#chatlog=chatlog[chatlog['intentname']!='greetings']
chatlog.head()

#useful columns - 'conversationreq', 'intentname', 'product', 'dunsid' - extract is for 13th oct
chatlog = chatlog[['agentid','conversationreq','conversationid','conversationmessageid', 
                   'con_request_time', 'con_response_time','conversationres','Duration','intentname', 'product', 'dunsid','requesttimestamp',
                   'responsetimestamp', 'syear', 'smonth']] 
chatlog['intentname']=chatlog['intentname'].fillna('') 
#Intent cleanse as per EIVA rules
arrApp= ['book_appointment','confirm_appointment','request_appointment_slots',
         'retrieve_appointment','req_updates_and_appointment','reject_appointment',
         'update_appointment','appointment']

# chatlog.head()
chatlog.info()


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
chatlog.loc[chatlog['product'].isna(), 'product'] = chatlog.groupby(['conversationid'])['product'].transform(lambda x: x.mode()[0] if any(x.mode()) else '')

chatlog['product'].fillna('',inplace=True)

# 2
chatlog['conversationreq'].fillna('',inplace=True)

# 3 convert numeric only cells with na eg: '324242423' as ''
chatlog['conversationreq_processed'] = chatlog['conversationreq'] .apply(lambda item : '' if str(item).isdigit() else item)

#4 process converse_req to remove punctuation and stop words
punctuations = '[!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~]' 
chatlog['conversationreq']=chatlog['conversationreq_processed'].apply(str) 
#chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(text_process)
chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].str.replace('[{}]'.format(punctuations), ' ')  #use punctuations and not string.punctuationfor '_' ; drop punctuations and stop words

chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(str) 

#5 Remove numbers from a string eg: 'Track order 123123' as 'Track order'
chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(lambda x: re.sub('\d', '', str(x)))
       
#add a full stop to request to help during join condition in agent page
chatlog['conversationreq']=chatlog['conversationreq'].apply(lambda x: x if x.endswith('.') else x + '.' )
#6 Connect Desk Agent, Track Order - join with '_' in  conversationreq_processed
chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(lambda x: x.strip().replace(r" ","_") if x.strip().lower()=='connect desk agent' else x)
chatlog['conversationreq_processed']=chatlog['conversationreq_processed'].apply(lambda x: x.strip().replace(r" ","_") if x.strip().lower().startswith('track order') else x)


#7 len of original request string
chatlog['conversationreq_len']=chatlog['conversationreq'].apply(str).apply(len)
chatlog.conversationreq_processed.head(50)


#preprocessing block

vader = SentimentIntensityAnalyzer()
chatlog['sentiment'] = chatlog['conversationreq'].apply(lambda row: get_sentiment(row)) 
chatlog['sentimenttext']=chatlog['sentiment'].apply(lambda x: 'Positive' if x>0.05 else('Negative' if x<-0.05 else 'Neutral')  )


chatlog['conversationres']=chatlog['conversationres'].astype(str)
chatlog['conversationres_processed']=chatlog['conversationres'].apply(lambda x: (x.split('bubbleText')[-1]) if 'bubbleText' in x else '')

#n=1 gives first instance of the split
chatlog['conversationres_processed']=chatlog['conversationres_processed'].str.split(',',expand = True,n=1)
#have res_processed for Rocketchat, make it null for eiva
chatlog['conversationres_processed']=chatlog[['conversationres_processed','agentid']].apply(lambda x: '' if x.agentid =='Eiva' else x.conversationres_processed,axis=1 )

#there is still \ to be replace do it later please
chatlog['conversationres_processed']=chatlog['conversationres_processed'].apply(lambda x: x.replace(r":",""))

#where the rockecthat agent is not available- replace wait with queue please
chatlog['wait'] = chatlog['conversationres'].apply(lambda x: "True" if "Current Avg Wait time in queue is" in x else "False" )
#chatlog['wait_time'] =chatlog['','wait'].applyt(lambda x" " if 'True' , axis=1) 
chatlog['dunsid'] = chatlog[['conversationid','dunsid']].groupby(['conversationid'], sort=False)['dunsid'].apply(lambda x: x.ffill().bfill())

# #convert dunsid to string
duns['dunsid'] = duns['dunsid'].fillna(0).apply(int)
duns['dunsid'] = duns['dunsid'].apply(str)
duns=duns[duns['sm_disabled_flag']=='Active']
duns.head()

dunsunique=duns[['dunsid','cp_name']].drop_duplicates()

# #join chat and duns to get CP name
chatlog =pd.merge(chatlog,dunsunique ,how='left',on='dunsid')
chatlog['cp_name'].fillna('', inplace=True) 
# chatlog.head()
chatlog.info()

p='(OR011-\d+|OR01\d+|OR02\d+|OGEA\d+|OS01\d+|OS02\d+|OFVA\d+|SGEA\d+|LLSS\d+|STSS\d+|LMSS\d+|0141\d+|LLCM\d+|1-\d+|2-\d+|5-7-\d+|5-8-\d+|A\d+|A0\d+|A1\d+|B0\d+|S0\d+|A5\d+|A6\d+|A9\d+|B1\d+|S1\d+|A2\d+|ONP\d+|PRH\d+|UUB\d+|NUN\d+|QBJ\d+|LLS\d+|FBC\d+|LLL\d+|LUE\d+|PAR\d+|UFB\d+|MFB\d+|FBM\d+|TEZ\d+|FBW\d+|KBB\d+|OOF\d+|NEN\d+|FBL\d+|LLEA\d+|LLEM\d+|EVZ\d+|LYE\d+|UWR\d+|UNB\d+|FBN\d+|FBS\d+|LYW\d+|FBT\d+|OOZ\d+|MTU\d+|OGE\d+|TBP\d+|UWB\d+|LNC\d+|LTL\d+|OFP\d+|UBB\d+|OZP\d+|LKC\d+|QRD\d+|PWH\d+|OVP\d+|LLWS\d+|LLWW\d+|LLWN\d+|LLLW\d+|LLWM\d+|LLWE\d+|LLWR\d+|OVZ\d+|OHP\d+|NNX\d+|TUZ\d+|FBE\d+|LLM\d+|LWE\d+|LTE\d+|LLN\d+|OZZ\d+|PPR\d+|QSJ\d+)'

chatlog['valid_order']=chatlog['conversationreq'].str.extract(p, expand=True)

chatlog['valid_order'] = chatlog[['conversationid','valid_order']].groupby(['conversationid'], sort=False)['valid_order'].apply(lambda x: x.ffill().bfill())

chatlog['valid_order'].fillna('',inplace=True)

i, r = pd.factorize(chatlog.valid_order)

choices = np.arange(max(99999, r.size))

c = np.random.choice(choices, r.shape, False)

chatlog=chatlog.assign(mask_order=c[i])

chatlog['mask_order']=chatlog.apply(lambda x : '' if x['valid_order'] =='' else x['mask_order'],axis=1 )


chatlog['Rocketchat']= chatlog[['conversationid','agentid']].groupby('conversationid')['agentid'].transform(lambda x: 'True' if x.nunique()>1 else 'False')

chatlog['conversationres']=chatlog['conversationres'].astype(str)

chatlog['conversationres_processed']=chatlog['conversationres'].apply(lambda x: (x.split('bubbleText')[-1]) if 'bubbleText' in x else '')

 

#retain only alpha, numbers and space

full_pattern = re.compile('[^a-zA-Z0-9 .]')

chatlog['conversationres_processed']= chatlog['conversationres_processed'].apply(lambda x: re.sub(full_pattern, "", x))

 

 

# replace 'usecaseContext' with ''

chatlog['conversationres_processed']=chatlog['conversationres_processed'].apply(lambda x: x.strip().replace(r"usecaseContext",""))

chatlog['conversationres_processed']=chatlog['conversationres_processed'].apply(lambda x: x.strip().replace(r"msgTypenormal",""))

chatlog['conversationres_processed']=chatlog['conversationres_processed'].apply(lambda x: x.strip().replace(r"msgTypeaccepted",""))

 

#add a full stop to response_processed to help during join condition in agent page

chatlog['conversationres_processed']=chatlog['conversationres_processed'].apply(lambda x: x if x.endswith('.') else x + '.' )

 

#have res_processed for Rocketchat, make it null for eiva

chatlog['conversationres_processed']=chatlog[['conversationres_processed','agentid']].apply(lambda x: '' if x.agentid =='Eiva' else x.conversationres_processed,axis=1 )





# random agent names 
agentlist=['AA','BB','CC','DD','EE','FF','GG','HH','II','JJ','KK','LL','MM','NN','OO','PP','QQ','RR','SS','TT', 'UU','VV','WW']
c = chatlog['conversationid'].unique()

vals = np.random.choice(agentlist, size=len(c))
chatlog['agentname'] = chatlog[chatlog['agentid']=='liveagent']['conversationid'].map(dict(zip(c, vals)))
chatlog['agentname'].fillna('',inplace=True)


#wait ends at which row
chatlog['wait_end'] = chatlog['conversationres'].apply(lambda x: "True" if "has joined the Conversation" in x else "False" )

dfrocketchat =chatlog[['agentid','conversationid','agentname','conversationreq','conversationreq_processed',
                      'conversationres_processed','cp_name','product','Duration','wait','wait_end','con_request_time']]

dfrocketchat = dfrocketchat[dfrocketchat['agentid']=='liveagent']
dfrocketchat.drop('agentid',axis=1, inplace=True)
dfrocketchat.info()

# revised_res = chatlog[~chatlog['conversationres_processed'].str.contains('|'.join(exclist))]['conversationres_processed']

#groupby to derice, concatenated response,concatenated request, HT sum(),
dfrocketchat_converged = dfrocketchat.assign(**
                                            {'conversationreq_join': dfrocketchat.groupby(by=['conversationid'])['conversationreq'].transform(lambda x: ' '.join(x)),
                                             'conversationres_processed_join': dfrocketchat.groupby(by=['conversationid'])['conversationres_processed'].transform(lambda x: ' '.join(x)),                                               
                                             'convHT_s': dfrocketchat.groupby(by=['conversationid'])['Duration'].transform(lambda x: sum(x)),
                                             'convwaittime_s':
                                             dfrocketchat[(dfrocketchat['wait']=='True') | (dfrocketchat['wait_end']=='True')]
                                             .groupby(by=['conversationid'])['con_request_time'].transform(lambda x: x.diff())
                                             })

#extract only the total_seconds() from convwaittime_s
dfrocketchat_converged.convwaittime_s= dfrocketchat_converged.convwaittime_s.apply(lambda x : x.total_seconds())

#fill duplicate rows with NaN convwaittime_s as existing valid convwaittime_s
dfrocketchat_converged['convwaittime_s'] = dfrocketchat_converged.groupby(['conversationid'], sort=False)['convwaittime_s'].apply(lambda x: x.bfill().ffill())
dfrocketchat_converged['convwaittime_s'].fillna(0,inplace=True)

dfrocketchat_converged.drop(['conversationreq','conversationreq_processed','conversationres_processed','Duration','wait','wait_end','con_request_time'], axis=1, inplace=True)
dfrocketchat_converged.drop_duplicates(inplace=True)
dfrocketchat_converged['FCR'] = dfrocketchat_converged['conversationres_processed_join'].apply(lambda x: 'True' if "is there anything else i can help with" in x.lower() else 'False')
dfrocketchat_converged.head()


dfrocketchat_converged['cust_sentiment_score'] = dfrocketchat_converged['conversationreq_join'].apply(lambda row: get_sentiment(row)) 
dfrocketchat_converged['agent_sentiment_score'] = dfrocketchat_converged['conversationres_processed_join'].apply(lambda row: get_sentiment(row)) 

dfrocketchat_converged['cust_sentiment']=dfrocketchat_converged['cust_sentiment_score'].apply(lambda x: 'Positive' if x>0.05 else('Negative' if x<-0.05 else 'Neutral')  )
dfrocketchat_converged['agent_sentiment']=dfrocketchat_converged['agent_sentiment_score'].apply(lambda x: 'Positive' if x>0.05 else('Negative' if x<-0.05 else 'Neutral')  )
dfrocketchat_converged['con_request_time']=chatlog['con_request_time']

from lorem_text import lorem

words = 10

 

dfrocketchat_converged['req_mask']=''

dfrocketchat_converged['res_mask']=''

dfrocketchat_converged['req_mask']=dfrocketchat_converged['req_mask'].apply(lambda x:lorem.words(words))

dfrocketchat_converged['res_mask']=dfrocketchat_converged['res_mask'].apply(lambda x:lorem.words(words))


i, r = pd.factorize(dfrocketchat_converged.conversationid)

choices = np.arange(max(999999, r.size))

c = np.random.choice(choices, r.shape, False)

dfrocketchat_converged=dfrocketchat_converged.assign(mask_convid=c[i])

#print(chatlog[['conversationid']].nunique())

#print(chatlog[['mask_convid']].nunique()) 




def aspect_table(req_dict):
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    dfaspectreq = chatlog_[chatlog_['conversationreq_processed']!='']['conversationreq_processed']
    
    #important : replace _ with '' as count vectorixer splits words with _ as 2 separate words
    #dfaspectreq= dfaspectreq['conversationreq_processed'].apply(lambda x: x.strip().replace(r"_",""))

     

    count_vectorizer = CountVectorizer(strip_accents ='unicode', lowercase=True,stop_words='english',token_pattern=r'\b[a-zA-Z]{4,}\b')

    count_data = count_vectorizer.fit_transform(dfaspectreq)  ########imp without word tokenize fit_transorm() works,word_tokenize is to get noun, adj etc!

     

    #print("--- %s count vectorizer---" % (time.time() - start_time))

     

     

    #start_time = time.time()

    words = count_vectorizer.get_feature_names()

    #print("--- %s feature names---" % (time.time() - start_time))

     

    total_counts = np.zeros(len(words))

    for t in count_data:

                    total_counts+=t.toarray()[0]

     

    #start_time = time.time()

    count_dict = (zip(words, total_counts))

    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:20] #### top 20 words

    words = [w[0] for w in count_dict]

    counts = [int(w[1]) for w in count_dict]

    dfaspect=pd.DataFrame({'Aspect':np.array(words),'Count':np.array(counts)})
    dfaspect=dfaspect[(dfaspect['Aspect']!='chatbotclosedmanually')  & (dfaspect['Aspect']!='idlesessiontimeoutclose')] 
    dfaspect=dfaspect.to_dict('records')
    return dfaspect


# fig,ax=plt.subplots(1,1,figsize=(12,8))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,8))

# ax[0].title.set_text ='Chat count per CP'
# ax[1].title.set_text ='Order count per CP'

dfcp=chatlog[['conversationid','cp_name']].groupby(['conversationid','cp_name']).count().reset_index()
print(dfcp.head())

# nms.dropna(thresh=2)
dfcpo=chatlog.dropna(how='any',subset=['valid_order'])
dfcpo=dfcpo[['conversationid','cp_name']].groupby(['conversationid','cp_name']).count().reset_index()

# ax.set_xticklabels(dfcp['cp_name'], rotation=90, ha='right')

dfcp['cp_name'].value_counts().head(20).plot(kind="bar",rot=90,color='purple',ax=ax1)
ax1.set_title ('Chat count per CP')
# ax1.axis("tight")

dfcpo['cp_name'].value_counts().head(20).plot(kind="bar",rot=90,color='cyan',ax=ax2)
ax2.set_title('Order count per CP')

# for p in ax[0].patches:
#              ax[0].annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
#                  ha='center', va='center', fontsize=11, color='black', xytext=(0, 20),
#                  textcoords='offset points')
        
# for p in ax[1].patches:
#              ax[1].annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
#                  ha='center', va='center', fontsize=11, color='black', xytext=(0, 20),
#                  textcoords='offset points')        



fig.suptitle('Volume Analysis', fontsize=25)


def order_table():

    test2= chatlog[['conversationid','Duration','valid_order']]

    test2 =test2[test2['valid_order'].notnull()==True]

    final=test2.assign(**{'chat_TotalDuration': test2.groupby(by=['valid_order'])['Duration'].transform(lambda x: x.sum()),
                        'Chat_session_count': test2.groupby(by=['valid_order'])['conversationid'].transform(lambda x: x.nunique())
                         })  
    final['chat_TotalDuration'] = final['chat_TotalDuration'].astype(str).str[-18:-10]
    final=final[['valid_order','chat_TotalDuration','Chat_session_count']]
    final=final.drop_duplicates().sort_values('chat_TotalDuration', ascending=False).head(20)
    final=final.to_dict('records')
    return final

def bar_chart_sentiment(req_dict):
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    #fig,ax=plt.subplots(1,1,figsize=(10,8))
    #plt.title("EIVA intent versus sentiment")
    dfintentsentimentEiva= chatlog_
    dfintentsentimentEiva = dfintentsentimentEiva[['EIVAIntent', 'sentimenttext']]
    dfintentsentimentEiva=dfintentsentimentEiva.groupby(['EIVAIntent', 'sentimenttext']).agg('size').sort_values(ascending=False).unstack()
    dfintentsentimentEiva=dfintentsentimentEiva.fillna(0)
    dfintentsentimentEiva['Total'] = dfintentsentimentEiva['Negative'] + dfintentsentimentEiva['Neutral'] + dfintentsentimentEiva['Positive']

    #print(dfintentsentimentEiva.sort_values('Total',ascending=False).head())
    #dfintentsentimentEiva.sort_values('Total').drop('Total',axis=1).plot.barh(stacked=True,ax=ax)
    dfintentsentimentEiva=dfintentsentimentEiva.sort_values('Total', ascending=False)
    intents=[]
    negative=[]
    neutral=[]
    positive=[]
    for i in range(0,6):
           intents.append(dfintentsentimentEiva.index[i])
           negative.append(dfintentsentimentEiva.Negative[i])
           neutral.append(dfintentsentimentEiva.Neutral[i])
           positive.append(dfintentsentimentEiva.Positive[i]) 
    dfintentsentimentEiva={'Intents': intents,
                      'Negative': negative,
                      'Neutral': neutral,
                      'Positive': positive}   
    return dfintentsentimentEiva

def sentimentcp(req_dict):

    #fig,ax=plt.subplots(1,1,figsize=(10,8))
    #plt.title("EIVA intent versus sentiment")
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    dfintentsentimentEiva= chatlog_
    dfintentsentimentEiva = dfintentsentimentEiva[['cp_name', 'sentimenttext']]
    dfintentsentimentEiva=dfintentsentimentEiva.groupby(['cp_name', 'sentimenttext']).agg('size').sort_values(ascending=False).unstack()
    dfintentsentimentEiva=dfintentsentimentEiva.fillna(0)
    dfintentsentimentEiva['Total'] = dfintentsentimentEiva['Negative'] + dfintentsentimentEiva['Neutral'] + dfintentsentimentEiva['Positive']

    #print(dfintentsentimentEiva.sort_values('Total',ascending=False).head())
    #dfintentsentimentEiva.sort_values('Total').drop('Total',axis=1).plot.barh(stacked=True,ax=ax)
    dfintentsentimentEiva=dfintentsentimentEiva.sort_values('Total', ascending=False)
    cp=[]
    negative=[]
    neutral=[]
    positive=[]
    for i in range(0,6):
           cp.append(dfintentsentimentEiva.index[i])
           negative.append(dfintentsentimentEiva.Negative[i])
           neutral.append(dfintentsentimentEiva.Neutral[i])
           positive.append(dfintentsentimentEiva.Positive[i]) 
    dfintentsentimentEiva1={'Cp': cp,
                      'Negative': negative,
                      'Neutral': neutral,
                      'Positive': positive}   
    return dfintentsentimentEiva1


def sentimentproduct(req_dict):
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    #fig,ax=plt.subplots(1,1,figsize=(10,8))
    #plt.title("EIVA intent versus sentiment")
    dfintentsentimentEiva= chatlog_
    dfintentsentimentEiva = dfintentsentimentEiva[['product', 'sentimenttext']]
    dfintentsentimentEiva=dfintentsentimentEiva.groupby(['product', 'sentimenttext']).agg('size').sort_values(ascending=False).unstack()
    dfintentsentimentEiva=dfintentsentimentEiva.fillna(0)
    dfintentsentimentEiva['Total'] = dfintentsentimentEiva['Negative'] + dfintentsentimentEiva['Neutral'] + dfintentsentimentEiva['Positive']

    #print(dfintentsentimentEiva.sort_values('Total',ascending=False).head())
    #dfintentsentimentEiva.sort_values('Total').drop('Total',axis=1).plot.barh(stacked=True,ax=ax)
    dfintentsentimentEiva=dfintentsentimentEiva.sort_values('Total', ascending=False)
    product=[]
    negative=[]
    neutral=[]
    positive=[]
    for i in range(0,6):
           product.append(dfintentsentimentEiva.index[i])
           negative.append(dfintentsentimentEiva.Negative[i])
           neutral.append(dfintentsentimentEiva.Neutral[i])
           positive.append(dfintentsentimentEiva.Positive[i]) 
    dfintentsentimentEiva2={'Product': product,
                      'Negative': negative,
                      'Neutral': neutral,
                      'Positive': positive}   
    return dfintentsentimentEiva2 


def topbar_values(req_dict):
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    #chatlog = pd.read_csv(r'C:\Users\612890877\Desktop\Wordmeter\conversationlog_Train_2.csv',encoding = "ISO-8859-1", low_memory=False)
    Totalsessioncount = chatlog_['conversationid'].nunique()
    cpcount=chatlog[['cp_name']].nunique()
    cpcount=cpcount.values
    cpcount=cpcount[0]
    prodcount=chatlog_[['product']].dropna(how='any',subset=['product']).nunique()
    prodcount=prodcount.values
    prodcount=prodcount[0]
    ordcount=chatlog_[['valid_order']].dropna(how='any',subset=['valid_order']).nunique()
    ordcount=ordcount.values
    ordcount=ordcount[0]
    Rocketsessioncount = chatlog_[chatlog_['agentid']=='liveagent']['conversationid'].nunique()
    dftemp=chatlog_
    
    dftemp['wait'] = dftemp['conversationres'].apply(lambda x: "True" if "Current Avg Wait time in queue is" in x else "False" )

    Rocketchatholdcount=dftemp [dftemp ['wait']=='True'][['conversationid']].nunique()
    Rocketchatholdcount=Rocketchatholdcount.values
    Rocketchatholdcount=Rocketchatholdcount[0]
  
    AHT=chatlog_[chatlog_['agentid'] =='liveagent'].groupby('conversationid')['Duration'].sum().reset_index()

    AHT=round(AHT['Duration'].sum()/AHT.shape[0]/60,0)
    #AHT=AHT[0]
    chatlog_['conversationres']=chatlog_['conversationres'].astype(str)

    chatlog_['FCR'] = chatlog_['conversationres'].apply(lambda x: "True" if "is there anything else i can help with" in x.lower() else "False" )
    FCRcount=chatlog_[chatlog_['FCR']=='True'][['conversationid']].drop_duplicates().shape[0] 
    RCCount=chatlog_[chatlog_['agentid']=='liveagent']['conversationid'].nunique()
    FCR=(round((FCRcount/RCCount)*100,2))  
    Queue= (round((Rocketchatholdcount/RCCount)*100,2))   
    topbar={'TotalSessionCount': str(Totalsessioncount),
             'Cpcount': str(cpcount),
             'Prodcount': str(prodcount),
             'Ordcount': str(ordcount),
             'Rocketsessioncount': str(Rocketsessioncount),
             'Rocketchatholdcount':str(Queue),
             'AHT':str(AHT),
             'FCR':str(FCR)
            }   
    return topbar
    
def doughnutchart_user(req_dict):  
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    dfcpsentiment = chatlog_[['conversationid','conversationmessageid','conversationreq','conversationreq_processed','agentid']]

    dfcpsentiment=dfcpsentiment.groupby(['conversationid','agentid'])['conversationreq'].apply(lambda x: ' '.join(x)).reset_index()  #separator as space

    dfcpsentiment['sentiment_compound'] = dfcpsentiment['conversationreq'].apply(lambda row: get_sentiment(row))



    #convert the compund values to sentiment text

    dfcpsentiment['sentimenttext']=dfcpsentiment['sentiment_compound'].apply(lambda x: 'Positive' if x>0.05 else('Negative' if x<-0.05 else 'Neutral')  )

    dfcpsentiment.head()



    size_of_groups_all =dfcpsentiment['sentimenttext'].value_counts().tolist()

    valuecntlabels_all=dfcpsentiment['sentimenttext'].value_counts().index.tolist()
    doughnutdata={"Labels":valuecntlabels_all,
          "Values":size_of_groups_all
        
    }
    return doughnutdata    

def Agentbarchart(req_dict):
    df=req_dict
    date=df['date']
    dfrocketchat_converged_=dfrocketchat_converged[dfrocketchat_converged['con_request_time'].dt.day==date]
    dfagentscore=dfrocketchat_converged_[['agentname', 'agent_sentiment']]
    dfagentscore = dfagentscore.assign(**{'Total': dfrocketchat_converged_.groupby('agentname')['agentname'].transform(lambda x: x.count()),
                           'positive': dfrocketchat_converged_.groupby('agentname')['agent_sentiment'].
                           transform(lambda x: x[x == 'Positive'].count())
                          })
    dfagentscore.drop('agent_sentiment',inplace=True,axis=1)
    dfagentscore.drop_duplicates(inplace=True)

    dfagentscore['score'] = round((dfagentscore['positive']/dfagentscore['Total']),2)
    Agentname=[]
    Scores=[]
    agentname=dfagentscore['agentname'].reset_index(drop=True)
    scores=dfagentscore['score'].reset_index(drop=True)
    for i in range(0,len(agentname)):
        Agentname.append(agentname[i])
        Scores.append(scores[i])

    agentbardata1={"Labels":Agentname,
              "Values":Scores
                   }      
    return agentbardata1   

def Agenttree(req_dict):
    df=req_dict
    date=df['date']
    dfrocketchat_converged_=dfrocketchat_converged[dfrocketchat_converged['con_request_time'].dt.day==date]
    dfagenttree = dfrocketchat_converged_.groupby('agentname')['conversationid'].count().reset_index()
    dfagenttree['agentname']=dfagenttree['agentname'].reset_index(drop=True)
    dfagenttree['conversationid']=dfagenttree['conversationid'].reset_index(drop=True)
    dfagenttree=dfagenttree.to_dict()
    return dfagenttree

def neg_scoretable(req_dict):
    df=req_dict
    date=df['date']
    dfrocketchat_converged_=dfrocketchat_converged[dfrocketchat_converged['con_request_time'].dt.day==date]
    negative_review_table=dfrocketchat_converged_[(dfrocketchat_converged_['agent_sentiment']=='Negative') ][['mask_convid',
    'agentname',
    'cp_name',
    'product',
    'req_mask',
    'res_mask',
    'agent_sentiment_score',
    'agent_sentiment']] 
    negative_review_table=negative_review_table.reset_index(drop=True)
    negative_review_table=negative_review_table.replace(np.nan, '', regex=True)
    negative_review_table=negative_review_table.to_dict('records')
    
    return negative_review_table
    
def new_orderstable(req_dict):
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
    dforder= chatlog_[['conversationid','Duration','mask_order','cp_name','product','agentid']]

    dforder =dforder[(dforder['mask_order']!='' ) & (dforder['mask_order']!='A55' ) &

                     (dforder['mask_order']!='A1' ) &  (dforder['mask_order']!='A20' ) &

                    (dforder['mask_order']!='2-10' ) ]

     

    dfordersummary=dforder.assign(**{'TotalDuration_seconds': dforder.groupby(by=['mask_order'])['Duration'].transform(lambda x: x.sum()),

                        'session_count': dforder.groupby(by=['mask_order'])['conversationid'].transform(lambda x: x.nunique()),

    #                      'productlist': dforder[['mask_order','product']].groupby(['mask_order'])['product'].transform(lambda x: list(x)),

    #                      'cplist': dforder.groupby(['mask_order'])['cp_name'].transform(lambda x: list(x)),

    #                      'agentid': dforder.groupby(['mask_order'])['agentid'].transform(lambda x: list(x)),                                 

                                     

                         }) 

     

     

     

    dfordersummary=dfordersummary[['mask_order','session_count','TotalDuration_seconds','product', 'agentid','cp_name']].drop_duplicates()  #not needed chat_TotalDuration_seconds

    dfordersummary=dfordersummary.drop_duplicates()  #not needed chat_TotalDuration_seconds

     

     

    dfordersummary['product'].fillna('', inplace=True)

    dfordersummary=dfordersummary.groupby(['mask_order','TotalDuration_seconds','session_count']).agg({

                                                                                'cp_name':lambda x: set(list(x)),

                                                                                'product':lambda x: set(list(x)),

                                                                                'agentid':lambda x: set(list(x))

                                                                               }).reset_index()

     

    dfordersummary=dfordersummary.sort_values('session_count',ascending=False)
    dfordersummary=dfordersummary.astype(str)
    dfordersummary=dfordersummary.to_dict('records')
    return dfordersummary    
    
def customer_table(req_dict):
    df=req_dict
    date=df['date']
    chatlog_=chatlog[chatlog['con_request_time'].dt.day==date]
   
    dfcp=chatlog_[['cp_name','conversationid','Rocketchat','Duration']]



    #sum the duration of each conversationid

    dfcp['TotalDuration_mm'] = dfcp.groupby(['conversationid'])['Duration'].transform('sum')/60



    dfcp=dfcp[['cp_name','conversationid','Rocketchat','TotalDuration_mm']]

    dfcp.drop_duplicates(inplace=True)



    dfcp.head()



    dfcp1=dfcp.assign(**{

        'Totalsessions': dfcp[['cp_name','conversationid']].groupby(by=['cp_name'])['conversationid'].transform(lambda x: x.nunique()),

        'Rocketchat':dfcp[['Rocketchat','cp_name']].groupby(by=['cp_name'])['Rocketchat'].transform(lambda x: (x=='True').sum()),  

        'AHT_mm':dfcp[['TotalDuration_mm','cp_name']].groupby(by=['cp_name'])['TotalDuration_mm'].transform(lambda x: round(x.mean(),2)),  

        'MxHT_mm':dfcp[['TotalDuration_mm','cp_name']].groupby(by=['cp_name'])['TotalDuration_mm'].transform(lambda x: round(x.max())),  

        'MnHT_mm':dfcp[['TotalDuration_mm','cp_name']].groupby(by=['cp_name'])['TotalDuration_mm'].transform(lambda x: round(x.min())),  

        'P95HT_mm':dfcp[['TotalDuration_mm','cp_name']].groupby(by=['cp_name'])['TotalDuration_mm'].transform(lambda x: round(x.quantile(0.95))),  

    })



    dfcp1=dfcp1[['cp_name','Totalsessions','Rocketchat','AHT_mm','MxHT_mm','MnHT_mm','P95HT_mm']]

    dfcp1.drop_duplicates(inplace=True)



    dfcp1['TransferRate'] = round(dfcp1['Rocketchat']/dfcp1['Totalsessions'] *100,2)



    dfcp1 =dfcp1[['cp_name','Totalsessions','Rocketchat','TransferRate','AHT_mm','MxHT_mm','MnHT_mm','P95HT_mm']]



    dfcp1=dfcp1.sort_values('Totalsessions', ascending=False)
    dfcp1=dfcp1.to_dict('records')
    return dfcp1 

def trends_barchart(req_dict):
    data=req_dict
    user_input=data['filter']
    user_input_filter=data['Search']


    dfcp=chatlog.copy()

    dfcp=dfcp.set_index('con_request_time')
    dfcp['date'] = dfcp.index.date

    if user_input=='cp_name':
        col='cp_name'    
    elif user_input=='product':
        col='product'
    elif user_input=='intent':
        col='EIVAIntent'
    elif user_input=='agent':    
        col='agentname'
    elif user_input=='order':        
        col='valid_order'


    dfcp=dfcp[['date',col,'conversationid']]
    dfcp=dfcp.drop_duplicates()

    dfcp=dfcp[['date',col]]
    # dfcp.columns=['date','volume']
    dfcp=dfcp.dropna(how='any',subset=['date'])

    ###added 8th jan
    dfcp.reset_index(inplace=True)
    dfcp=dfcp.drop('con_request_time',axis=1)

    filter=''    
    if (user_input_filter)!='':
        if user_input=='cp_name':
            filter=data['Search']
            dfcp=dfcp[dfcp[col]==filter] ###############filter on a cp
        elif user_input=='product':
            filter=data['Search']
            dfcp=dfcp[dfcp[col]==filter] ###############filter on a cp    
        elif user_input=='intent':
            filter=data['Search']
            dfcp=dfcp[dfcp[col]==filter] ###############filter on a cp
        elif user_input=='agent':
            filter=data['Search']
            dfcp=dfcp[dfcp[col]==filter] ###############filter on a cp
        elif user_input=='order':
            filter=data['Search']
    #         dfcp=dfcp[dfcp[col]==filter] ###############filter on a cp

        test=dfcp.groupby('date').agg('count').reset_index()    
    else:
        test=dfcp.groupby('date')[col].nunique().reset_index()

    ###added 8th jan
    test['date']=test['date'].astype(str)
    test['date'][0]='13-10-2020'
    test['date'][1]='14-10-2020'
    test['date'][2]='15-10-2020'
    test=test.to_dict()
    return test
    
def trends_linechart(req_dict):
    data=req_dict
    user_input=data['filter']
    user_input_filter=data['Search']
    dfcp=chatlog.copy()

    dfcp=dfcp.dropna(how='any',subset=['con_request_time'])






    dfcp=dfcp[dfcp['conversationreq']!='']



    dfcp['con_request_time'] = dfcp.groupby(['conversationid']).con_request_time.transform('min')
    



    dfcp=dfcp[['cp_name','product','EIVAIntent','agentname','conversationid','conversationreq','conversationres_processed','con_request_time']]



    # dfcp=dfcp[dfcp['cp_name']=='TALKTALK COMMUNICATIONS LTD'] ###CP level

    filter=''

    if (user_input_filter)!='':

        if user_input=='cp_name':

    #         filter='TALKTALK COMMUNICATIONS LTD'

            filter=data['Search']

            dfcp=dfcp[dfcp['cp_name']==filter] ###############filter on a cp

        elif user_input=='product':

            filter=data['Search']

            #         filter='LineManageProvide'

            dfcp=dfcp[dfcp['product']==filter] ###############filter on a cp   

        elif user_input=='intent':

            filter=data['Search']

            dfcp=dfcp[dfcp['EIVAIntent']==filter] ###############filter on a cp

        elif user_input=='agent':   

            filter=data['Search']

            dfcp=dfcp[dfcp['agentname']==filter] ###############filter on a cp



    # print('filter ', user_input_filter)



    if user_input=='agent':

        dfcpsentiment=dfcp.groupby(['conversationid','con_request_time'])['conversationres_processed'].apply(lambda x: ' '.join(x)).reset_index()  #separator as space

        dfcpsentiment['sentiment_compound'] = dfcpsentiment['conversationres_processed'].apply(lambda row: get_sentiment(row))

    else:   

        dfcpsentiment=dfcp.groupby(['conversationid','con_request_time'])['conversationreq'].apply(lambda x: ' '.join(x)).reset_index()  #separator as space

        dfcpsentiment['sentiment_compound'] = dfcpsentiment['conversationreq'].apply(lambda row: get_sentiment(row))



    dfcpsentiment=dfcpsentiment.set_index('con_request_time')

    dfcpsentiment.head()



    # dfcpsentiment['sentiment_compound'] = dfcpsentiment['conversationreq'].apply(lambda row: get_sentiment(row))

    dfcpsentiment['sentimenttext']=dfcpsentiment['sentiment_compound'].apply(lambda x: 'Positive' if x>0.05 else('Negative' if x<-0.05 else 'Neutral')  )





    # dfcpsentiment=dfcpsentiment[['sentimenttext','sentiment_compound']]

    dfcpsentiment=dfcpsentiment[['sentimenttext']]





    dfcpsentiment=dfcpsentiment.reset_index()

    dfcpsentiment['con_request_time'] = dfcpsentiment['con_request_time'].apply(lambda x: x.replace(minute=0, second=0))
    dfcpsentiment['con_request_time'] = dfcpsentiment['con_request_time'].dt.strftime('%d/%m/%Y %H:%M')





    dfcpsentimentplot=dfcpsentiment.groupby(['con_request_time','sentimenttext']).agg('size').sort_values(ascending=False).unstack()

    dfcpsentimentplot= dfcpsentimentplot.reset_index()

    dfcpsentimentplot=dfcpsentimentplot.set_index('con_request_time')

    dfcpsentimentplot.fillna(0,inplace=True)

    # print(dfcpsentimentplot)



    dfneg=pd.DataFrame()

    dfpos=pd.DataFrame()

    dfneu=pd.DataFrame()



    if 'Negative' in dfcpsentimentplot.columns:

        dfneg=dfcpsentimentplot['Negative']



    if 'Positive' in dfcpsentimentplot.columns:   

        dfpos=dfcpsentimentplot['Positive']



    if 'Neutral' in dfcpsentimentplot.columns:   

        dfneu=dfcpsentimentplot['Neutral']
    df = pd.concat([dfneg,dfpos,dfneu],axis=1,sort=False).reset_index()
    df['con_request_time']=df['con_request_time'].astype(str)
    num=len(df['con_request_time'])
    datetime=[]
    positive=[]
    negative=[]
    neutral=[]
    for i in range(0,num):
            datetime.append(df['con_request_time'][i])
            positive.append(df['Positive'][i])
            negative.append(df['Negative'][i])
            neutral.append(df['Neutral'][i])
            i=i+1
            
    df={'datetime':datetime,
        'positive':positive,
        'negative':negative,
        'neutral':neutral,
       }
    return df    