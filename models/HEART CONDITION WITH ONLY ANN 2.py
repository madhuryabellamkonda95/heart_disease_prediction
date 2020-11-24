#!/usr/bin/env python
# coding: utf-8

# In[254]:


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


# In[255]:


data=pd.read_csv(r'C:\Users\DELLPC\Desktop\HEART CONDITION PROJECT\HeartCondition\datasets_heart_condition.csv')
print(data.head())

print(data.columns)

print('\n')

print(data.dtypes)


# In[256]:


print(data.isnull().sum())
data.dropna()
print(data.duplicated(subset= (['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])))


# In[257]:


data['target'].value_counts()


# In[258]:


data.shape


# In[259]:


#histograms of dataset

#the plots where we get only 2 bars or 3 are represented as categorical variables and rest as 
#continuous varibles
plt.figure(figsize=(2,2))
data.hist()


# In[260]:


#histograms of particula

plt.figure(figsize=(2,2))
data.cp.hist()


# In[261]:


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


# In[262]:


plt.figure(figsize=(10,5))
data.boxplot()


# In[263]:


data.describe()


# In[285]:


#distribution of the data with and without heart disease
#i.e with the target

plotting=sns.countplot(x='target' ,data=data)
plotting.set_title('HEART DISEASE DISTRIBUTION')

#labels for sticking for x axis
plotting.set_xticklabels(['heart disease absent','heart disease present'])

# the data provides info that no.of people having heart disease and without heart disease are at the same level


# In[265]:


#distribution of data with respect to gender(male and female) having and not having heart disease

plotting1=sns.countplot(x='target',data=data,hue='sex')

plotting1.set_title('HEART DISEASE DISTRIBUTION BASED ON GENDER')

plotting1.legend(['Female','Male'])

plotting1.set_xticklabels(['heart disease present','heart disease absent'])


# data provides an interpretation that number of both males and females having or not having heart disease are a

#at the same level

# equal number of males are having high chance of heart disease and also not having heart disease than females


# In[266]:


#plt.figure(figsize=(50, 50))
plt.rcParams["figure.figsize"] = (50,50)
heat_map=sns.heatmap(data.corr(method='pearson'),annot=True,fmt='.2f',linewidth=2,linecolor='black',square=True,annot_kws={'fontsize': 30})
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45,fontsize=50)


#no values =1 so no strong corelations


# In[267]:


#creating training and test data

X=np.array(data.drop(['target'],1))

print(X.shape)
y=np.array(data['target'])
print(y.shape)


# In[268]:


#standaradising by z score formula: (X-mean)/std

mean=X.mean(axis=0)
X-=mean
std=X.std(axis=0)
X/=std


# In[269]:


X[0]


# In[270]:


#train and test split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=43,test_size=0.2)


# In[271]:


y_train.shape
print(y_train[:10])


# In[272]:


#converting y _train and y_tset into categorical
from keras.utils.np_utils import to_categorical

Y_train=to_categorical(y_train)
Y_test=to_categorical(y_test)

print(Y_train.shape)
print(Y_train[:10])
print('\n')
print(X_train[0])


# In[273]:


y_train[y_train>0]=1
y_test[y_test>0]=1

print(y_train[:20])


# In[274]:


Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# In[275]:


#creating model and training the model



def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())


# In[276]:


history=binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=100, batch_size=10)


# In[277]:


binary_model.save("heart_condition.h5")
print("Saved model to disk")


# In[278]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[279]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[280]:


from sklearn.metrics import classification_report, accuracy_score
# generate classification report using predictions for binary model 
binary_pred = np.round(binary_model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))


# In[281]:


from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score
print('CONFUSION MATRIX')

predictions=binary_model.predict(X_test)
predictions= (predictions > 0.5)

confmatrix=confusion_matrix(Y_test_binary,predictions)
print(confmatrix)


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




