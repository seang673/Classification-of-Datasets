#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


#Load in Datasets
training_data_4 = pd.read_csv("TrainData4.txt", delimiter='\s+')
training_label_4 = pd.read_csv("TrainLabel4.txt", delimiter='\s+')
test_data_4 = pd.read_csv("TestData4.txt", delimiter='\s+')


# In[7]:


training_data_4.head(10)


# In[8]:


test_data_4.head(20)


# In[10]:


training_label_4.head(10)


# In[13]:


#Variables to mark the missing values
missing_value_mark = 1.00000000000000e+99

#Use imputer object to fill missing values with the mean of column
imputer = SimpleImputer(missing_values= missing_value_mark, strategy='mean')
training_data_4_imputed = imputer.fit_transform(training_data_4)
test_data_4_imputed = imputer.fit_transform(test_data_4)


# In[14]:


#Standardize features to have a mean of 0 and a variance of 1
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(training_data_4_imputed)
test_data_scaled = scaler.transform(test_data_4_imputed)


# In[16]:


#Turn training label into 1D array
train_label_array = training_label_4.values.ravel()


# In[19]:


#Using Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data_scaled, train_label_array)


# In[20]:


test_labels_prediction = classifier.predict(test_data_scaled)


# In[21]:


print("Predicted Test Labels: ", test_labels_prediction)


# In[24]:


# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(train_data_scaled, train_label_array, test_size=0.2, random_state=42)

# Train the classifier on the training split
classifier.fit(X_train_split, y_train_split)

# Make predictions on the validation split 
y_val_pred = classifier.predict(X_val_split) 

# Evaluate the accuracy accuracy = accuracy_score(y_val_split, y_val_pred) 
print('Validation Accuracy:') 
# Print detailed classification report 
print(classification_report(y_val_split, y_val_pred, zero_division=1))


# In[ ]:




