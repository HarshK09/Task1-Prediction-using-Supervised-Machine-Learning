#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised Machine Learning
# 
# **Problem Statement**
# 
# 1) Predict the percentage of an student based on the number of study hours.
# 
# 2) This is a simple linear regression task as it involves just 2 variables.
# 
# 3) What will be predicted score if a student studies for 9.25 hrs/day?
# 
# **Author: Harsh Kukretee** 
# 
# **GRIP @ The Sparks Foundation**

# **Step 1: Imported all the necessary libraries required for this task**
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# **Reading data from URL**

# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


sns.scatterplot(x=df['Hours'],y=df['Scores'])


# In[8]:


sns.regplot(x=df['Hours'],y=df['Scores'])


# # Sepearting feature(s) and target

# In[9]:


x=df['Hours']
y=df['Scores']


# # Train-Test Split

# In[10]:


train,test = train_test_split(df,test_size=0.25,random_state=0)


# In[11]:


train_x=train.drop("Scores",axis=1)
train_y=train["Scores"]


# In[12]:


test_x=test.drop("Scores",axis=1)
test_y=test["Scores"]


# # Model Building

# In[13]:


regressor = LinearRegression()


# In[14]:


regressor.fit(train_x,train_y)


# In[15]:


# Plotting the regression line # formula for line is y=m*x + c
line = regressor.coef_*train_x+regressor.intercept_

# Plotting for the test data
plt.scatter(train_x,train_y)
plt.plot(train_x, line);
plt.show()


# # Making Predictions

# In[16]:


predictions = regressor.predict(test_x)


# In[17]:


list(zip(test_y,predictions))


# # Q. What will be predicted score if a student studies for 9.25 hrs/ day? 

# In[18]:


hour =[9.25]
own_pr=regressor.predict([hour])
print("Number of Hours = {}".format([hour]))
print("Predicted Score = {}".format(own_pr[0]))

