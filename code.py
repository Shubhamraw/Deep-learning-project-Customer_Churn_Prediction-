#!/usr/bin/env python
# coding: utf-8

# Data set customer churn modeling in which a bank has given a fictional data based on these features we need to identify whether the customer is going to stay with the bank or to close the account and get away with the bank

# ### Following steps are taken for building the project
# 1. Importing libraries(tensorflow.keras,numpy,pandas,sklearn)
# 2. Reading the csv file and cleaning the data 
# 3. Splitting the data into train and test and standardizing Feature  
# 4. Building ANN --> Adding input layer, Random w init and Adding Hidden Layers with activation function
# 5. Select Optimizer, Loss, and Performance Metrics and Compiling the model
# 6. using model.fit to train the model
# 7. Prediction
# 8. Evaluate the model
# 9. Adjust optimization parameters or model if needed

# In[3]:


import tensorflow as tf
from tensorflow import keras #keras is embedded into tensorflow 2.0
from tensorflow.keras import Sequential #layers: list of layers to add to the model.
from tensorflow.keras.layers import Flatten, Dense #Just your regular densely-connected NN layer.


# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[5]:


data = pd.read_csv('Customer_Churn_Modelling.csv')


# In[6]:


data.head()


# In[8]:


X = data.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis=1)
y = data['Exited']


# In[9]:


X.head()


# ANN works on a numerical data not on a string data so we have to use map function to map this data into a numerical data (label encoder or one hot encoder)

# In[11]:


from sklearn.preprocessing import LabelEncoder 
# It can also be used to transform non-numerical labels 
# (as long as they are hashable and comparable) to numerical labels.


# In[23]:


label1 = LabelEncoder()
X['Geography'] = label1.fit_transform(X['Geography'])
X.head()
#0-france, 1-germany, 2-spain


# In[22]:


label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])
X.head() # 0-female and 1-male


# These are categorical values so we need to convert it into one hot encoding by using sklearn or we can use pandas get dummies

# In[24]:


X = pd.get_dummies(X, drop_first=True, columns=['Geography'])
X.head()


# ### Feature Standardisation

# In[25]:


from sklearn.preprocessing import StandardScaler #Standardize features by removing the mean and scaling to unit variance


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0, stratify = y)


# In[28]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train


# ### Building ANN

# In[29]:


model = Sequential()
model.add(Dense(X.shape[1], activation='relu', input_dim = X.shape[1]))
model.add(Dense(128, activation='relu')) # hidden layer
model.add(Dense(1, activation = 'sigmoid')) #two output
# if we do not apply a Activation function then the output signal would simply be a simple linear function.A linear function is just a polynomial of one degree


# In[31]:


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[34]:


model.fit(X_train, y_train.to_numpy(), batch_size = 10, epochs = 10, verbose =1)


# In[35]:


y_pred = model.predict_classes(X_test)


# In[36]:


y_pred


# In[37]:


y_test


# In[38]:


model.evaluate(X_test, y_test.to_numpy())


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[40]:


confusion_matrix(y_test, y_pred)


# In[41]:


accuracy_score(y_test, y_pred)


# In[ ]:




