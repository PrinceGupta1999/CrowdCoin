
# coding: utf-8

# In[72]:


# %matplotlib inline
# %reload_ext autoreload
# %autoreload 
import argparse
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from IPython.display import HTML, display


# In[73]:


train=pd.read_csv('hackNSUT/train.csv')
# test=pd.read_csv('hackNSUT/train.csv')


# In[82]:


import sys, json 
if len(sys.argv) >= 2:
    data  = json.loads(sys.argv[1])
else:
    data = {"data":["Apple Inc.","NSDQ","30-09-11",43818,108249,13331]}
test = pd.DataFrame()
test['comp_name'] = [data["data"][0]]
test['exchange'] = [data["data"][1]]
test['per_end_date'] = [data["data"][2]]
test['gross_profit'] = [data["data"][3]]
test['tot_revnu'] = [data["data"][4]]
# print(data["data"][1])
# test.head()


# In[74]:


train.isna().sum(axis=0)
test.isna().sum(axis=0)


# In[47]:


train=train[(train.isnull().sum(axis=1)==0)]
test=test[(test.isnull().sum(axis=1)==0)]


# In[48]:


x=train['per_end_date'];
xx=x.str.split('-',expand=True)
train['per_end_year']=xx[2]
train['per_end_month']=xx[1]
# train['Date']=train['Date'].astype(str)
# train['Month']=train['Month'].astype(str)
train.drop(labels='per_end_date',axis=1,inplace=True)


# In[49]:


x=test['per_end_date'];
xx=x.str.split('-',expand=True)
test['per_end_year']=xx[2]
test['per_end_month']=xx[1]
# train['Date']=train['Date'].astype(str)
# train['Month']=train['Month'].astype(str)
test.drop(labels='per_end_date',axis=1,inplace=True)


# In[50]:


train['per_end_year']=train['per_end_year'].astype(int)
train['per_end_month']=train['per_end_month'].astype(int)
train.head()


# In[51]:


test['per_end_year']=test['per_end_year'].astype(int)
test['per_end_month']=test['per_end_month'].astype(int)
test.head()


# In[52]:


train['gross_profit']=train['gross_profit'].astype(int)
train['tot_revnu']=train['tot_revnu'].astype(int)
train['comm_stock_net']=train['comm_stock_net'].astype(int)
train.head()


# In[53]:


test['gross_profit']=test['gross_profit'].astype(int)
test['tot_revnu']=test['tot_revnu'].astype(int)
# test['comm_stock_net']=test['comm_stock_net'].astype(int)

test.head()


# In[54]:


cat_vars=['comp_name','exchange','per_end_year','per_end_month']
contin_vars = ['gross_profit','tot_revnu','comm_stock_net']


# In[55]:


test['comm_stock_net']=0


# In[56]:


for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()


# In[57]:


for v in contin_vars:
    train[v] = train[v].fillna(0).astype('float32')


# In[58]:


for v in cat_vars: test[v] = test[v].astype('category').cat.as_ordered()


# In[59]:


for v in contin_vars:
    test[v] = test[v].fillna(0).astype('float32')


# In[60]:


df, y, nas, mapper = proc_df(train, 'comm_stock_net', do_scale=True)
yl = np.log(y)


# In[61]:


df_test, _, nas, mapper = proc_df(test, 'comm_stock_net', do_scale=True)
                                 # mapper=mapper, na_dict=nas)


# In[62]:


from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
df = scX.fit_transform(df)
# x_test = scX.transform(x_test)
df_test = scX.transform(df_test)


# In[63]:


filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))


# In[64]:


prediction=model.predict(df_test)


# In[65]:


print(prediction)


# In[42]:




