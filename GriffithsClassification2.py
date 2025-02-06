#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np 
import pandas as pd 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[6]:


#Load in the datasets
training_data_2 = pd.read_csv("TrainData2.txt", delimiter='\s+')
training_label_2 = pd.read_csv("TrainLabel2.txt", delimiter='\s+')
test_data_2 = pd.read_csv("TestData2.txt", delimiter='\s+')


# In[54]:


training_data_2.head(10)


# In[55]:


test_data_2.head(10)


# In[33]:


training_label_2.head(10)


# In[34]:


#Convert training data into a dataframe
train_df = pd.DataFrame(training_data_2)


# In[35]:


#Conver training data into a dataframe
test_df = pd.DataFrame(test_data_2)


# In[36]:


#Initialize and Perform KNN Imputation on missing values for training and test data
knn_imputer = KNNImputer(missing_values=1.00000000000000e+99, n_neighbors=5)
training_data_2_imputed = knn_imputer.fit_transform(train_df)
test_data_2_imputed =  knn_imputer.fit_transform(test_df)


# In[48]:


#Standardize features to have a mean of 0 and a variance of 1
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(training_data_2_imputed)
test_data_scaled = scaler.transform(test_data_2_imputed)


# In[49]:


#Turn training label into 1D array
train_label_array = training_label_2.values.ravel()


# In[50]:


#Using Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_data_scaled, train_label_array)


# In[52]:


test_labels_prediction = classifier.predict(test_data_scaled)


# In[53]:


print("Predicted Test Labels: ", test_labels_prediction)


# In[64]:


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




