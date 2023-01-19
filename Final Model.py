#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from pickle import dump
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('review_rating_1.csv',index_col=[0])
data


# In[3]:


data['Rating_Analysis'] = data['Rating_Analysis'].replace({'Negative': -1})
data['Rating_Analysis'] = data['Rating_Analysis'].replace({'Positive': 1})
data['Rating_Analysis'] = data['Rating_Analysis'].replace({'Neutral': 0})
data


# ## TF-ID Vectorizer

# In[5]:


count=TfidfVectorizer()
x_train_df=count.fit(data['Clean_Review'])


# In[6]:


x_train=x_train_df.transform(data['Clean_Review'])


# In[7]:


X_train,X_test,y_train,y_test=train_test_split(x_train,data['Rating_Analysis'],shuffle=True,random_state=30,test_size=0.3)


# In[8]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[9]:


def Accuracy(y_train,y_train_pred,y_test,y_test_pred):
    print('Train Accuracy\n')
    print(classification_report(y_train,y_train_pred))
    print('\n',confusion_matrix(y_train,y_train_pred))
    print('\n',accuracy_score(y_train,y_train_pred))
    print('*'*100)
    print('Test Accuracy\n')
    print(classification_report(y_test,y_test_pred))
    print('\n',confusion_matrix(y_test,y_test_pred))
    print('\n',accuracy_score(y_test,y_test_pred)) 


# ## Final model 

# In[10]:


from lightgbm import LGBMClassifier


# In[11]:


LGBM = LGBMClassifier()


# In[12]:


LGBM.fit(X_train, y_train)


# In[13]:


LGBM_train=LGBM.predict(X_train)
LGBM_test=LGBM.predict(X_test)


# In[14]:


LGBM_model=Accuracy(LGBM_train,y_train,LGBM_test,y_test)
LGBM_model


# ## deployment

# In[15]:


# build the intelligence for tfid Vectorizer 
x=data['Clean_Review']
y=data['Rating_Analysis']


# In[16]:


tfid=TfidfVectorizer()
tfid_deploy=tfid.fit(x)


# In[17]:


# converting text into numeric for svm
x_train=tfid_deploy.transform(x)


# In[18]:


LGBM_model_deploy=LGBMClassifier()
LGBM_model_deploy.fit(x_train,y)


# In[19]:


#saving svm n tfid into pkl
dump(obj=LGBM_model_deploy,file=open('LGBM_model_deploy.pkl','wb'))
dump(obj=tfid_deploy,file=open('tfid_deploy.pkl','wb'))


# In[ ]:




