#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


#Load in Datasets
training_data_5 = pd.read_csv("TrainData5.txt", delimiter='\s+')
training_label_5 = pd.read_csv("TrainLabel5.txt", delimiter='\s+')
test_data_5 = pd.read_csv("TestData5.txt", delimiter='\s+')


# In[6]:


training_data_5.head(10)


# In[7]:


test_data_5.head(10)


# In[14]:


training_label_5.head(10)


# In[11]:


#Variables to mark the missing values
missing_value_mark = 1.00000000000000e+99

#Use imputer object to fill missing values with the mode of column
imputer = SimpleImputer(missing_values= missing_value_mark, strategy='most_frequent')
training_data_5_imputed = imputer.fit_transform(training_data_5)
test_data_5_imputed = imputer.fit_transform(test_data_5)


# In[12]:


#Standardize features to have a mean of 0 and a variance of 1
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(training_data_5_imputed)
test_data_scaled = scaler.transform(test_data_5_imputed)


# In[13]:


training_label_array = training_label_5.values.ravel()


# In[19]:


#Use KNN classifier with 5 neighbors on training data and training label
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data_scaled, training_label_array)


# In[21]:


test_labels_prediction = knn.predict(test_data_scaled)


# In[22]:


print("Predicted Test Labels: ", test_labels_prediction)


# In[25]:


# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(train_data_scaled, training_label_array, test_size=0.2, random_state=42)

# Train the classifier on the training split
knn.fit(X_train_split, y_train_split)

# Make predictions on the validation split 
y_val_pred = knn.predict(X_val_split) 

# Evaluate the accuracy accuracy = accuracy_score(y_val_split, y_val_pred) 
print('Validation Accuracy:') 
# Print detailed classification report 
print(classification_report(y_val_split, y_val_pred, zero_division=1))


# In[ ]:




