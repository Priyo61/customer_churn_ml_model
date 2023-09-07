#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[10]:


df=pd.read_csv('C:\\Users\\Priyo\\OneDrive\\Desktop\\cus.csv')
df.head()


# In[11]:


df.dtypes


# In[12]:


df.shape


# In[14]:


df.isnull().sum()


# In[18]:


df[df.duplicated()]


# In[19]:


df.head(2)


# In[122]:


import matplotlib.pyplot as plt
plt.hist(df['Customer service calls'] , 20)
plt.xlabel("number of services")
plt.ylabel("number of customers")
plt.title('Total number of cutomers in all services')


# In[132]:


plt.hist(df['Total day charge'] , 100);
plt.xlabel('total charges')
plt.ylabel('count')
plt.show();


# In[136]:


plt.hist(df['Account length'] , 20);


# In[ ]:





# In[ ]:





# In[21]:


df.dtypes


# In[30]:


df.pop('Unnamed: 0')


# In[28]:


df.pop('State')


# In[44]:


df.head()


# In[38]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['n_inter_plan']=le.fit_transform(df['International plan'])
df['n_voice_plan']=le.fit_transform(df['Voice mail plan'])
df['n_churn']=le.fit_transform(df['Churn'])


# In[47]:


df.pop('Voice mail plan')


# In[48]:


df.pop('International plan')


# In[51]:


df.pop('Churn')


# In[60]:


df.head(3)


# In[54]:


df.shape


# In[65]:


indep=df.drop('n_churn',axis=1)
dep=df['n_churn']
dep
indep.head(2)


# In[64]:


dep.value_counts()


# In[71]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
new_in=sc.fit_transform(indep)
new_in


# In[87]:


new_in=pd.DataFrame(new_in)
new_in.describe()


# In[84]:


import seaborn as sns
sns.heatmap(new_in.corr(),annot=True)


# In[89]:


new_in.shape


# In[90]:


dep.shape


# In[91]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(new_in,dep,test_size=0.25)


# In[93]:


from sklearn.linear_model import LogisticRegression
mod=LogisticRegression()


# In[94]:


mod.fit(x_train,y_train)


# In[95]:


predic=mod.predict(x_test)


# In[100]:


final=pd.DataFrame({'Actual':y_test,'Prediction':predic})
final=pd.concat((new_in,final),axis=1)

final = final[final["Actual"]>=0]

final


# In[105]:


from sklearn.metrics import accuracy_score
accuracy_score(final['Actual'],final['Prediction'])


# In[106]:


from sklearn.tree import DecisionTreeClassifier
decision_model = DecisionTreeClassifier()
decision_model.fit(x_train, y_train)


# In[107]:


prediction = decision_model.predict(x_test)
accuracy_score(y_test,prediction)


# In[108]:


from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier()
                                            
                                    

RandomForest_model.fit(x_train, y_train)


# In[109]:


prediction_r = RandomForest_model.predict(x_test)
accuracy_score(y_test,prediction_r)


# In[110]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5)

knn_model.fit(x_train, y_train)


# In[111]:


prediction_knn = knn_model.predict(x_test)
accuracy_score(y_test,prediction_knn)


# In[112]:


get_ipython().system('pip install xgboost')


# In[113]:


from xgboost import XGBClassifier
XGB_model = XGBClassifier()

XGB_model.fit(x_train, y_train)


# In[114]:


prediction_xgb = XGB_model.predict(x_test)
accuracy_score(y_test,prediction_xgb)


# In[115]:


from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(x_train, y_train)


# In[116]:


prediction_svc = svc_model.predict(x_test)
accuracy_score(y_test,prediction_svc)

