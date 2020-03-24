#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# In[8]:


iris_dataset = load_iris()
print("Target Names: {}".format(iris_dataset['target_names']))
print("Feature name: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(iris_dataset['data']))


# In[3]:


print("Shape of data: {}".format(iris_dataset['data'].shape))


# In[4]:


# Data set is already Label Encoded.
# setosa -> 0
# versicolor -> 1
# virginica -> 2

print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target :\n {}".format(iris_dataset['target']))


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
print("X_train shape :{}".format(X_train.shape))
print("y_train shape :{}".format(y_train.shape))


# In[9]:


print("X_test shape:{}".format(X_test.shape))
print("Y_test shape:{}".format(y_test.shape))


# In[10]:


knn = KNeighborsClassifier(n_neighbors = 1)

# Building our model on the training set using fit method
knn.fit(X_train, y_train)


# In[11]:


# Taken a sample input
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new shape : {}".format(X_new.shape))


# In[12]:


prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))


# In[13]:


print(iris_dataset['target_names'])


# In[16]:


# Running the model on test set and checking it's accuracy
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == y_test)))


# In[17]:


print("Test set score (knn.score) : {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




