#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
train_file = os.path.normpath('C:\\projekte\\Santander\\Santander_Customer_Transaction_Prediction\\Data\\train.csv')
test_file = os.path.normpath('C:\\projekte\\Santander\\Santander_Customer_Transaction_Prediction\\Data\\test.csv')


# In[14]:


import pandas as pd
df_train_file = pd.read_csv(train_file)
print(df_train_file.info)


# In[7]:


df_test_file = pd.read_csv(test_file)


# In[8]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


# In[29]:


X_Train_File = df_train_file.drop(['ID_code','target'],axis=1)
Y_Train_File = df_train_file['target']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_Train_File,Y_Train_File,test_size=0.2,shuffle=True,stratify=Y_Train_File)


# In[30]:


print("Shape of Training data : ",x_train.shape)
print("Shape of validation data :",x_test.shape)


# In[31]:


estimator = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',max_iter=1000,verbose=1).fit(X_Train, y_Train)


# In[32]:


print(estimator.score(x_train,y_train))


# In[33]:


print(estimator.score(x_test,y_test))


# ### Simple estimator result

# In[45]:


X_test_file = df_test_file.drop(['ID_code'],axis=1)
y_test_predicted = estimator.predict(X_test_file)


# In[43]:


df_output = pd.concat([df_test_file['ID_code'],pd.DataFrame(y_test_predicted,columns=['target'])],axis=1)
df_output.to_csv('C:\projekte\Santander\Santander_Customer_Transaction_Prediction\Data\predicted.csv',index=False)


# #### Kaggle output result = 0.62913

# In[52]:


model = SelectFromModel(estimator, prefit=True)
X_new = model.transform(x_train)
print(X_new.shape)


# model = SelectFromModel(estimator, prefit=False)
# model.fit(x_train,y_train)

# In[68]:


x_new = model.get_support(indices=True)
print(x_new)
print("total number of features = ",len(x_new))


# In[72]:


x_new = model.transform(x_train)
print(x_new.shape)


# In[74]:


logreg_feat_selected = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',max_iter=1000,verbose=1).fit(x_new, y_train)


# In[76]:


print(logreg_feat_selected.score(x_new,y_train))


# In[77]:


print(logreg_feat_selected.score(model.transform(x_test),y_test))


# In[79]:


x_test_transformed = model.transform(X_test_file)
y_predicted = logreg_feat_selected.predict(x_test_transformed)


# In[80]:


df_output = pd.concat([df_test_file['ID_code'],pd.DataFrame(y_predicted,columns=['target'])],axis=1)
df_output.to_csv('C:\projekte\Santander\Santander_Customer_Transaction_Prediction\Data\predicted_logreg_feat_sel.csv',index=False)


# #### Kaggle score : 0.51139

# # Random Forest

# In[85]:


from sklearn.ensemble import RandomForestClassifier as RF_Classifier
clf_rf = RF_Classifier(n_estimators=100,verbose=2)


# In[86]:


clf_rf.fit(x_train,y_train)


# In[87]:


clf_rf.score(x_train,y_train)


# In[89]:


clf_rf.score(x_test,y_test)


# In[90]:


rf_predicted = clf_rf.predict(X_test_file)


# In[91]:


df_output = pd.concat([df_test_file['ID_code'],pd.DataFrame(rf_predicted,columns=['target'])],axis=1)
df_output.to_csv('C:\projekte\Santander\Santander_Customer_Transaction_Prediction\Data\predicted_rf.csv',index=False)

