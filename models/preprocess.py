#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import packages that we will be working with.
import os
import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns

np.random.seed(10)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers


# In[ ]:


data=pd.read_csv(r'C:\Users\DELLPC\Desktop\HEART CONDITION PROJECT\HeartCondition\datasets_heart_condition.csv')
print(data.head())

print(data.columns)

print('\n')

print(data.dtypes)


# In[ ]:


print(data.isnull().sum())
data.dropna()
print(data.duplicated(subset= (['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])))


# In[ ]:


data['target'].value_counts()


# In[ ]:


data.shape


# In[ ]:


#histograms of dataset

#the plots where we get only 2 bars or 3 are represented as categorical variables and rest as 
#continuous varibles
plt.figure(figsize=(2,2))
data.hist()


# In[ ]:


#histograms of particula

plt.figure(figsize=(2,2))
data.cp.hist()


# In[ ]:


#4 outliers
plt.figure(figsize=(45,5))

sns.boxplot(data["age"])
print('\n')

sns.boxplot(data["chol"])

print('\n')

sns.boxplot(data["thal"])

print('\n')

sns.boxplot(data["trestbps"])
plt.show()
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
data.boxplot()


# In[ ]:


data.describe()


# In[ ]:


#distribution of the data with and without heart disease
#i.e with the target

plotting=sns.countplot(x='target' ,data=data)
plotting.set_title('HEART DISEASE DISTRIBUTION')

#labels for sticking for x axis
plotting.set_xticklabels(['heart disease absent','heart disease present'])

# the data provides info that no.of people having heart disease and without heart disease are at the same level


# In[ ]:


#distribution of data with respect to gender(male and female) having and not having heart disease

plotting1=sns.countplot(x='target',data=data,hue='sex')

plotting1.set_title('HEART DISEASE DISTRIBUTION BASED ON GENDER')

plotting1.legend(['Female','Male'])

plotting1.set_xticklabels(['heart disease present','heart disease absent'])


# data provides an interpretation that number of both males and females having or not having heart disease are a

#at the same level

# equal number of males are having high chance of heart disease and also not having heart disease than females


# In[ ]:


#plt.figure(figsize=(50, 50))
plt.rcParams["figure.figsize"] = (50,50)
heat_map=sns.heatmap(data.corr(method='pearson'),annot=True,fmt='.2f',linewidth=2,linecolor='black',square=True,annot_kws={'fontsize': 30})
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45,fontsize=50)


#no values =1 so no strong corelations


# In[ ]:


#creating training and test data

X=np.array(data.drop(['target'],1))

print(X.shape)
y=np.array(data['target'])
print(y.shape)


# In[ ]:


#standaradising by z score formula: (X-mean)/std

mean=X.mean(axis=0)
X-=mean
std=X.std(axis=0)
X/=std


# In[ ]:


#train and test split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=43,test_size=0.2)


# In[ ]:


#converting y _train and y_tset into categorical
from keras.utils.np_utils import to_categorical

Y_train=to_categorical(y_train)
Y_test=to_categorical(y_test)

print(Y_train.shape)
print(Y_train[:10])
print('\n')
print(X_train[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




