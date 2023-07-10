#!/usr/bin/env python
# coding: utf-8

# # ML PROJECT: College Applications

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[2]:


data_set = pd.read_csv("D:\College_Admissions.csv",encoding ="ISO-8859-1",engine="python")


# In[3]:


data_set.rename({'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'}, axis=1, inplace=True)
data_set.drop('Serial No.', axis=1, inplace=True)


# In[4]:


data_set.head()


# ### Checking For Null Values

# In[5]:


data_set.isna().any()


# ### Checking for duplicate values

# In[6]:


data_set.duplicated()


# In[7]:


data_set


# In[8]:


data_set.info()


# ### Checking the mean, standard deviation, minimum and maximum value for each feature

# In[9]:


data_set.describe()


# ### Dropping duplicates( As we have none in this specific data set, it doesn't make a difference)

# In[10]:


data_set.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)


# ### Analyzing features from the data set

# In[11]:


sns.histplot(x = 'TOEFL Score' ,data = data_set, color = 'blue').set_title('Distributions for CGPA')


# In[12]:


sns.relplot(y=data_set['TOEFL Score'],x=data_set['Chance of Admit'], color = 'red')


# In[13]:


sns.histplot(x = 'GRE Score' ,data = data_set, color = 'blue').set_title('Distributions for GRE')


# In[14]:


sns.relplot(y=data_set['GRE Score'],x=data_set['Chance of Admit'], color = 'Red')


# In[15]:


sns.histplot(x = 'CGPA' ,data = data_set, color = 'blue').set_title('Distributions for CGPA')


# In[16]:


sns.relplot(y=data_set['CGPA'],x=data_set['Chance of Admit'], color= 'red')


# In[17]:


sns.pairplot(data_set,hue = "Chance of Admit",palette='bwr')


# ### Calculating correlation for each feature wrt target variable

# In[18]:


correlation = pd.DataFrame(data_set.corr()['Chance of Admit'])
correlation.sort_values(['Chance of Admit'], ascending=False, inplace = True)
correlation


# ### Normalizing Features 

# In[19]:


def normalizeFeatures(X):
    mu=np.mean(X)
    sigma=np.std(X)
    X_norm=(X - mu)/sigma
    return X_norm


# In[20]:


data_set['GRE Score'] = normalizeFeatures(data_set['GRE Score'])
data_set['TOEFL Score'] = normalizeFeatures(data_set['TOEFL Score'])
data_set['University Rating'] = normalizeFeatures(data_set['University Rating'])
data_set['LOR'] = normalizeFeatures(data_set['LOR'])
data_set['SOP'] = normalizeFeatures(data_set['SOP'])
data_set['CGPA'] = normalizeFeatures(data_set['CGPA'])
data_set['Research'] = normalizeFeatures(data_set['Research'])


# ### Splitting the data set into training and testing set

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X = data_set[['GRE Score', 'TOEFL Score', 'University Rating','LOR', 'SOP','CGPA', 'Research']]
y=  data_set['Chance of Admit']


# In[23]:


X_train, X_test, y_train,y_test = train_test_split(X,y , random_state = 105,  train_size=0.8, shuffle=True)


# ### Scaling the code using standard scaler

# In[24]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ### Importing linear regression model and fitting the training data

# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


ft = LinearRegression()


# In[27]:


ft.fit(X_train, y_train)


# In[28]:


predictions = ft.predict(X_test)


# ### Calculating Mean square error

# In[29]:


mean_squared_error(y_test,predictions)


# In[30]:


ft.score(X_test,y_test)


# ### Performing PCA on the data set

# In[31]:


from sklearn.decomposition import PCA


# In[32]:


admission = pd.read_csv("D:\College_Admissions.csv")
admission.head()
print(len(admission))


# In[33]:


pca = PCA(n_components=2)


# In[34]:


X_pca = pca.fit_transform(X)


# In[35]:


A_train, A_test, b_train,b_test = train_test_split(X_pca ,y , random_state = 105, test_size=0.2, shuffle=True)
sc = StandardScaler()


# In[36]:


A_train = sc.fit_transform(X_train)
A_test = sc.fit_transform(X_test)


# ### Training the regression model with PCA data set

# In[37]:


gt = LinearRegression()
gt.fit(A_train, b_train)


# In[38]:


predictions_PCA = gt.predict(A_test)


# In[39]:


mean_squared_error(y_test,predictions_PCA)


# In[40]:


gt.score(A_test,b_test) 


# In[41]:


print(b_test)
print(predictions_PCA)


# ### Regression analysis

# In[42]:


fig = sns.regplot(x = y, y = X_pca[:,0], ci =None, scatter_kws={"color": "blue"}, line_kws={"color": "black"}).set(title = 'Regression Line' ,ylabel='PCA Column 1')


# In[43]:


fig = sns.regplot(x = y, y = X_pca[:,1], ci =None, scatter_kws={"color": "blue"}, line_kws={"color": "black"}).set(title = 'Regression Line' ,ylabel='PCA Column 2')


# ### Copying the predictions of the models into CSV

# In[44]:


#Saving predictions in CSV file
from numpy import savetxt
savetxt('LR_Predictions.csv', predictions, delimiter=',', fmt = '%.2f')


# In[45]:


from numpy import savetxt
savetxt('LR_PCA_Predictions.csv', predictions_PCA, delimiter=',', fmt = '%.2f')
  

