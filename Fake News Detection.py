#!/usr/bin/env python
# coding: utf-8

# Data downloaded from https://www.kaggle.com/c/fake-news/data?select=test.csv
# 
# train.csv: A full training dataset with the following attributes:
# 
#     id: unique id for a news article
#     title: the title of a news article
#     author: author of the news article
#     text: the text of the article; could be incomplete
#     label: a label that marks the article as potentially unreliable
#         1: unreliable
#         0: reliable
# 

# In[328]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[329]:


fake = pd.read_csv('news/Fake.csv')
true = pd.read_csv('news/True.csv')


# In[330]:


fake.head()


# In[331]:


fake.info()


# In[332]:


true.head()


# In[333]:


true.info()


# ### Cleaning

# In[334]:


#track fake vs true news
true['target'] = 'true'
fake['target'] = 'false'


# In[335]:


#Combine both dataframes
df_combined = pd.concat([true, fake])
df_combined.head()


# In[336]:


df_combined.target.value_counts()


# In[ ]:





# In[337]:


df_combined['total_text'] = df_combined['title'] + ' ' + df_combined['text']


# In[338]:


#Search for null values
df_combined[df_combined['total_text'].isnull()]


# In[339]:


#Remove columns that won't be used
df_combined = df_combined.drop(['title', 'text', 'date'], axis = 1)
df_combined.head()


# In[340]:


# Convert to lowercase
df_combined['total_text'] = df_combined['total_text'].str.lower()
df_combined


# In[341]:


# Remove punctuation
df_combined['total_text'] = df_combined['total_text'].str.replace('[^\w\s]','')
df_combined


# In[342]:


df_combined[df_combined['total_text'].str.contains("us")]


# In[343]:


#Remove stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

df_combined['total_text'] = df_combined['total_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[344]:


df_combined['total_text'].head(20)


# In[345]:


#Tokenize total_text column
from nltk import tokenize
nltk.download('punkt')
from nltk.tokenize import WhitespaceTokenizer

df_combined['tokenized']=df_combined['total_text'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize)


# In[346]:


df_combined.head()


# In[347]:


#Lemmatize text
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text] 

df_combined['lemmatized'] = df_combined['tokenized'].apply(lemmatize_text)


# In[348]:


#US has been lemmatized to u
df_combined.head()


# ### Exploration

# In[349]:


plt.subplots(figsize=(12,8))
df_combined['subject'].hist();


# In[350]:


df_combined['target'].hist();


# In[351]:


plt.subplots(figsize=(12,8))
sns.histplot(data=df_combined, x='subject', hue='target');


# In[ ]:





# In[ ]:




