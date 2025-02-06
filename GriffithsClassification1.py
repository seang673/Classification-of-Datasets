#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[13]:


#Load in the datasets
training_data_1 = pd.read_csv("TrainData1.txt", delimiter='\t')
training_label_1 = pd.read_csv("TrainLabel1.txt", delimiter='\t')
test_data_1 = pd.read_csv("TestData1.txt", delimiter='\t')


# In[14]:


#To show first 10 rows of TrainData1
training_data_1.head(10)


# In[15]:


#To show first 10 rows of TestData1
test_data_1.head(10)


# In[16]:


#Variables to mark the missing values
missing_value_mark = 1.00000000000000e+99

#Use imputer object to fill missing values with the mean of column
imputer = SimpleImputer(missing_values= missing_value_mark, strategy='mean')
training_data_1_imputed = imputer.fit_transform(training_data_1)
test_data_1_imputed = imputer.fit_transform(test_data_1)


# In[17]:


#Feature Scaling
scaler = StandardScaler()
training_data_1_scaled = scaler.fit_transform(training_data_1_imputed) 
test_data_1_scaled = scaler.transform(test_data_1_imputed)


# In[18]:


train_label_array = training_label_1.values.ravel()


# In[20]:


#Use SVM classifier on training data and training label
svm_classifier = SVC()
svm_classifier.fit(training_data_1_scaled, train_label_array)


# In[21]:


# Predict test labels 
test_labels_prediction = svm_classifier.predict(test_data_1_scaled)


# In[22]:


print("Predicted Test Labels: ", test_labels_prediction)


# In[27]:


# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(training_data_1_scaled, train_label_array, test_size=0.2, random_state=42)

# Train the classifier on the training split
svm_classifier.fit(X_train_split, y_train_split)

# Make predictions on the validation split 
y_val_pred = svm_classifier.predict(X_val_split) 

# Evaluate the accuracy accuracy = accuracy_score(y_val_split, y_val_pred) 
print('Validation Accuracy:') 
# Print detailed classification report 
print(classification_report(y_val_split, y_val_pred, zero_division=1))


# In[ ]:




