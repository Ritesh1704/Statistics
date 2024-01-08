#!/usr/bin/env python
# coding: utf-8

# Statistics: branch of mathematics that deals with collection, analyzing and interpreting the data, 
#     which helps us to get some inferences form the data.
# 
# 1) Descriptive stats : Helps in describing what is present in the data.
# 2) Inferential stats : helps us to draw some coclusions from the sample...Hypothesis testing, correlation 

# Descriptive stats:
# 1) Central tendencies: Mean, Median, Mode
# 2) Dispersions: 
#     Variation - spread of the data
#         -> Range
#         -> Quartiles - q1, q2,q3
#         -> Variance

# In[ ]:


Sampling:
    1) Random sampling
    2) Startified Sampling


# In[1]:


#import libraries
import pandas as pd


# In[2]:


df = pd.read_csv("country_profile_variables.csv")

#df = pd.read_csv(r"path of the dataset")


# In[3]:


df


# In[6]:


df.head() #top 5 rows


# In[7]:


df.tail()


# In[8]:


df.shape


# In[11]:


#fetch the column names
list(df.columns)


# In[13]:


df.info()


# In[14]:


#count of individual column
df.count()


# In[16]:


#statistical analysis
df.describe() #numerical data


# In[17]:


df.describe(include='all')


# In[19]:


df['country']


# In[21]:


#check null value
df.isnull()


# In[22]:


df.isnull().sum()


# In[ ]:


#dealing with null values

#df.dropna()
#df.fillna()


# In[ ]:


#df['country'].fillna(df['country'].mode()[0])


# In[23]:


df['country'].mode()


# In[24]:


df['country'].mode()[0]


# In[25]:


#In our initial analysis we have found that the data doesn't contain any null values.


# ### We will begin our initial Descritive Statistical Analysis
# 
# For our analysis, we will choose a 'Population in thousands (2017)' column. We can perform the similar operationson other columns. We will get the measure of central tendency and measure of spread of the data.

# In[26]:


df['Population in thousands (2017)']


# In[27]:


#mean

df['Population in thousands (2017)'].mean()


# In[28]:


#median

df['Population in thousands (2017)'].median()


# In[29]:


#mode

df['Population in thousands (2017)'].mode()


# In[30]:


df['Population in thousands (2017)'].value_counts()


# In[31]:


df['Population in thousands (2017)'].unique()


# In[32]:


df['Population in thousands (2017)'].nunique()


# In[33]:


#range --> max-min

print(df['Population in thousands (2017)'].max())
print(df['Population in thousands (2017)'].min())


# In[34]:


df['Population in thousands (2017)'].max() - df['Population in thousands (2017)'].min()


# In[35]:


#variance

df['Population in thousands (2017)'].var()


# In[36]:


#std

df['Population in thousands (2017)'].std()


# In[38]:


#Calculate 25, 50, 75 percentile (1,100)

import numpy as np

twe = np.percentile(df['Population in thousands (2017)'], 25) #Q1
fif = np.percentile(df['Population in thousands (2017)'], 50) #Q2
sev = np.percentile(df['Population in thousands (2017)'], 75) #Q3

print(twe, fif, sev)


# In[39]:


#Quartile ==> 1,2,3 (0,4)
#Quantile ==> (0 to 1)

Q1 = df['Population in thousands (2017)'].quantile(0.25)
Q2 = df['Population in thousands (2017)'].quantile(0.50)
Q3 = df['Population in thousands (2017)'].quantile(0.75)

print(Q1, Q2, Q3)


# In[40]:


#Interquartile range

IQR = Q3-Q1
IQR


# In[41]:


import seaborn as sns
sns.boxplot(df['Population in thousands (2017)'])


# In[42]:


import matplotlib.pyplot as plt
plt.hist(df['Population in thousands (2017)'])


# In[43]:


print(Q1, Q2, Q3)


# In[44]:


lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR


# In[45]:


print(lower_bound,upper_bound )


# In[46]:


df = df[(df['Population in thousands (2017)']> lower_bound) & (df['Population in thousands (2017)']<upper_bound) ] 


# In[47]:


import seaborn as sns
sns.boxplot(df['Population in thousands (2017)'])


# In[48]:


import matplotlib.pyplot as plt
plt.hist(df['Population in thousands (2017)'])


# In[49]:


#create a density plot
sns.distplot(df['Population in thousands (2017)'])


# In[50]:


df.columns


# In[52]:


sns.distplot(df['Sex ratio (m per 100 f, 2017)'])


# In[53]:


#Proability


# What is the probability that the Population in thousands(2017) is more than 50,000.

# In[58]:


data = pd.read_csv("country_profile_variables.csv")


# In[59]:


data['Population in thousands (2017)']


# In[60]:


data['population']= np.where(data['Population in thousands (2017)']> 50000, 1, 0)


# In[61]:


data['population']


# In[62]:


data['population'].value_counts()


# In[64]:


27/(202+27)


# In[67]:


len(data[data['population']==1])/ len(data['population'])


# ### Pandas Profiling

# In[ ]:


pip install pandas-profiling


# In[68]:


import pandas as pd
from pandas_profiling import ProfileReport


# In[69]:


data = pd.read_csv("country_profile_variables.csv")


# In[70]:


rep = ProfileReport(data)


# In[71]:


rep


# In[ ]:




