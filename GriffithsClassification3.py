#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[93]:


#Load in Datasets
training_data_3 = pd.read_csv("TrainData3.txt", delimiter='\s+')
training_label_3 = pd.read_csv("TrainLabel3.txt", delimiter='\s+')
test_data_3 = pd.read_csv("TestData3.txt", delimiter=',')


# In[94]:


#Display first few rows of TrainData3
training_data_3.head(10)


# In[95]:


#Display first few rows of TestData3
test_data_3.head(10)


# In[96]:


#Variables to mark the missing values
missing_value_mark = 1.00000000000000e+99

#Use imputer object to fill missing values with the median of column
imputer = SimpleImputer(missing_values= missing_value_mark, strategy='most_frequent')
training_data_3_imputed = imputer.fit_transform(training_data_3)
test_data_3_imputed = imputer.fit_transform(test_data_3)


# In[97]:


#Feature Scaling
scaler = StandardScaler()
training_data_3_scaled = scaler.fit_transform(training_data_3_imputed) 
test_data_3_scaled = scaler.transform(test_data_3_imputed)


# In[98]:


#Convert labels to array
training_labels_array = training_label_3.values.ravel()


# In[99]:


#Use KNN classifier with 5 neighbors on training data and training label
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(training_data_3_scaled, training_labels_array)


# In[100]:


test_labels_prediction = knn.predict(test_data_3_scaled)


# In[101]:


print("Predicted Test Labels: ", test_labels_prediction)


# In[102]:


# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(training_data_3_scaled, training_labels_array, test_size=0.2, random_state=42)

# Train the classifier on the training split
knn.fit(X_train_split, y_train_split)

# Make predictions on the validation split 
y_val_pred = knn.predict(X_val_split) 

# Evaluate the accuracy accuracy = accuracy_score(y_val_split, y_val_pred) 
print('Validation Accuracy:') 
# Print detailed classification report 
print(classification_report(y_val_split, y_val_pred, zero_division=1))


# In[ ]:




