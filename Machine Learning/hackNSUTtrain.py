
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')


# In[2]:


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)


# In[3]:


PATH='hackNSUT/'
train=pd.read_csv('hackNSUT/train.csv')


# In[4]:


train.isna().sum(axis=0)


# In[5]:


train=train[(train.isnull().sum(axis=1)==0)]


# In[6]:


from IPython.display import HTML, display


# In[7]:


train.head()


# In[8]:


# for t in tables: display(DataFrameSummary(t).summary())


# In[9]:


# train = tables


# In[10]:


# test['y']=0


# In[11]:


# train=pd.concat([train,test])


# In[12]:


x=train['per_end_date'];
xx=x.str.split('-',expand=True)
train['per_end_year']=xx[2]
train['per_end_month']=xx[1]
# train['Date']=train['Date'].astype(str)
# train['Month']=train['Month'].astype(str)
train.drop(labels='per_end_date',axis=1,inplace=True)


# In[13]:


train['per_end_year']=train['per_end_year'].astype(int)
train['per_end_month']=train['per_end_month'].astype(int)
train.head()


# In[14]:


# train.drop(labels=['Date','Month'],axis=1,inplace=True)


# In[15]:


# train['Depart_Time_Hour'] = pd.to_datetime(train.Dep_Time).dt.hour
# train['Depart_Time_Minutes'] = pd.to_datetime(train.Dep_Time).dt.minute
# train['Depart_Time']=train['Depart_Time_Minutes']+train['Depart_Time_Hour']*60
# train.drop(labels = 'Dep_Time', axis = 1, inplace = True)


# In[16]:


# train.reset_index(inplace=True)
# train.drop(labels = 'Depart_Time_Hour', axis = 1, inplace = True)
# train.drop(labels = 'Depart_Time_Minutes', axis = 1, inplace = True)
# train.head()


# In[17]:


# train['Arr_Time_Hour'] = pd.to_datetime(train.Arrival_Time).dt.hour
# train['Arr_Time_Minutes'] = pd.to_datetime(train.Arrival_Time).dt.minute
# train['Arr_Time']=train['Arr_Time_Minutes']+train['Arr_Time_Hour']*60
# train.drop(labels = 'Arrival_Time', axis = 1, inplace = True)
# train.drop(labels = 'Arr_Time_Hour', axis = 1, inplace = True)
# train.drop(labels = 'Arr_Time_Minutes', axis = 1, inplace = True)
# train.head()


# In[18]:


# def changed(test):
#     test = test.strip()
#     total=test.split(' ')
#     to=total[0]
#     hrs=(int)(to[:-1])*60
#     if((len(total))==2):
#       #t0=total[0]
#         mint=(int)(total[1][:-1])
#         hrs=hrs+mint
#     test=str(hrs)
#     return test
  
  
#     train['Duration']=train['Duration'].apply(changed)
#     train.head()


# In[19]:


# def stops(x):
#     if(x=='non-stop'):
#         x=str(0)
#     else:
#         x.strip()
#     stps=x.split(' ')[0]
#     x=stps
#     return x

# train['Total_Stops']=train['Total_Stops'].apply(stops)
# train.head()


# In[20]:


# train.reset_index(inplace=True)


# In[21]:


# import pandas as pd
# pd.options.mode.chained_assignment = None 
# for i in range(train.shape[0]):
#     if(train['Additional_Info'][i]=='No info'):
#         train['Additional_Info'][i]='No Info'
# train.head()


# In[22]:


# train.drop(labels=['Route'],axis=1,inplace=True)


# In[23]:


# train.head()


# In[24]:


# train.per_end_month.nunique()


# In[25]:


# for i in range(train.shape[0]):
#     if(train['Destination'][i]=='New Delhi'):
#         train['Destination'][i]='Delhi'
# train.head()


# In[26]:


# train.Destination.unique()


# In[27]:


# train.head()


# In[28]:


train['gross_profit']=train['gross_profit'].astype(int)
train['tot_revnu']=train['tot_revnu'].astype(int)
train['comm_stock_net']=train['comm_stock_net'].astype(int)
train.head()


# In[29]:


# train=train.drop('Additional_Info',axis=1)
# train.head()


# In[30]:


cat_vars=['comp_name','exchange','per_end_year','per_end_month']
contin_vars = ['gross_profit','tot_revnu','comm_stock_net']


# In[31]:


# joined_test=train.iloc[10682:]
# joined=train.iloc[:10682]


# In[32]:


for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()


# In[33]:


train.head()


# In[34]:


# apply_cats(joined_test, joined)
joined=train


# In[35]:


for v in contin_vars:
    joined[v] = joined[v].fillna(0).astype('float32')
#     joined_test[v] = joined_test[v].fillna(0).astype('float32')


# In[36]:


joined.head()


# In[37]:


df, y, nas, mapper = proc_df(joined, 'comm_stock_net', do_scale=True)
yl = np.log(y)


# In[38]:


# df_test, _, nas, mapper = proc_df(joined_test, 'Price', do_scale=True,
#                                   mapper=mapper, na_dict=nas)


# In[39]:


# train_ratio = 0.75
train_ratio = 0.9
n=len(train)
train_size = int(n*train_ratio); train_size
val_idx = list(range(train_size, n))


# In[40]:


# df=df.drop('index',axis=1)


# In[41]:


# df_test=df_test.drop('index',axis=1)


# In[42]:


# df=df.drop('Additional_Info',axis=1)
# df_test=df_test.drop('Additional_Info',axis=1)


# In[43]:


# df.comp_name.nunique()


# In[44]:


# def inv_y(a): return np.exp(a)

# def exp_rmspe(y_pred, targ):
#     targ = inv_y(targ)
#     pct_var = (targ - inv_y(y_pred))/targ
#     return math.sqrt((pct_var**2).mean())

# max_log_y = np.max(yl)
# y_range = (0, max_log_y*1.2)
# max_y=np.max(y)
# y_range= (0, max_y*2)


# In[45]:


# md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=16)


# In[46]:


# cat_sz = [(c, len(joined[c].cat.categories)+1) for c in cat_vars]


# In[47]:


# cat_sz


# In[48]:


# emb_szs = [(c, c) for _,c in cat_sz]


# In[49]:


# emb_szs


# In[50]:


# ??md.get_learner


# In[51]:


# m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
#                    0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
# m.summary()


# In[52]:


# def rmse(x,y): return math.sqrt(((x-y)**2).mean())


# In[53]:


# lr = 1e-3
# m.lr_find()


# In[54]:


# m.sched.plot(1)


# In[55]:


# lr=1e-3


# In[56]:


# m.fit(lr, 3, metrics=[exp_rmspe])


# In[57]:


# m.fit(lr, 10, metrics=[exp_rmspe], cycle_len=3)


# In[58]:


# m.fit(lr, 10, metrics=[exp_rmspe], cycle_len=3)


# In[59]:


# m.fit(lr, 20, metrics=[exp_rmspe], cycle_len=3)


# In[60]:


# m.fit(lr, 2, metrics=[exp_rmspe], cycle_len=3)


# In[61]:


# m.save('val0')


# In[62]:


# m.load('val0')


# In[63]:


# x,y_=m.predict_with_targs()


# In[64]:


# x


# In[65]:


# y_


# In[66]:


# x=np.exp(x)
# y_=np.exp(y_)


# In[67]:


# def rmsle(Y,YH):
#     sum=0
#     for y,yh in zip(Y,YH):
#         sum+=(np.log(y)-np.log(yh))**2
#     return (sum/(Y.shape[0]))**0.5


# In[68]:


# print(1-rmsle(x,y_))


# In[69]:


# exp_rmspe(x,y_)


# In[70]:


# pred_test=m.predict()


# In[71]:


# pred_test = np.exp(pred_test)


# In[72]:


# pred_test=pred_test.astype(int)


# In[73]:


# pred_test


# In[74]:


# submission=pd.DataFrame(data=pred_test,columns=['Price'])


# In[75]:


# submission


# In[76]:


# submission.to_excel("submission.xlsx",index=False)


# In[77]:


# len(y)


# In[78]:


import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df,y,random_state=10)
def rmsle(Y,YH):
    sum=0
    for y,yh in zip(Y,YH):
        sum+=(np.log(y)-np.log(yh))**2
    return (sum/(Y.shape[0]))**0.5


# In[79]:


from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
x_train = scX.fit_transform(x_train)
x_test = scX.transform(x_test)
# df_test = scX.transform(df_test)


# In[80]:


import xgboost as xgb
model=xgb.XGBRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# In[81]:


y_pred=model.predict(x_test)


# In[82]:


y_test


# In[83]:


y_pred


# In[84]:


# def rmse(x,y): return math.sqrt(((x-y)**2).mean())

# def print_score(m):
#     res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
#                 m.score(X_train, y_train), m.score(X_valid, y_valid)]
#     if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
#     print(res)


# In[85]:


from sklearn.ensemble import RandomForestRegressor
rr=RandomForestRegressor(n_estimators=80,min_samples_leaf=1, max_features=0.99, oob_score=True,n_jobs=-1)
# rr=RandomForestRegressor()
rr.fit(x_train,y_train)

y_pred=rr.predict(x_test)


# In[86]:


y_pred


# In[87]:


y_test


# In[89]:


rr.score(x_test,y_test)


# In[90]:


preds = np.stack([t.predict(x_test) for t in rr.estimators_])
preds[:,0], np.mean(preds[:,0]), y_test[0]


# In[91]:


plt.plot([metrics.r2_score(y_test, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


# In[91]:


# yy_pred=rr.predict(df_test)
# yy_pred=yy_pred.astype(int)
# submission=pd.DataFrame({'Price':yy_pred})
# submission.to_excel("HACKNEW.xlsx",index=False)


# In[92]:


# df1=pd.read_excel('HACKNEW.xlsx')
# df2=pd.read_excel('submission.xlsx')


# In[93]:


# min((df2-df1)['Price'])


# In[94]:


# submission=pd.DataFrame(data=(df1+5*df2)/5,columns=['Price'])
# submission.to_excel("final.xlsx",index=False)


# In[95]:


# import xgboost as xgb
# model=xgb.XGBRegressor()
# model.fit(df, y)
# y_pred=model.predict(x_test)
# print(1-rmsle(y_test,y_pred))


# In[96]:


# from sklearn.ensemble import GradientBoostingRegressor
# model= GradientBoostingRegressor()
# model.fit(df, y)
# y_pred=model.predict(x_test)
# print(1-rmsle(y_test,y_pred))


# In[97]:


# import lightgbm as lgb
# train_data=lgb.Dataset(x_train,label=y_train)
# params = {'learning_rate':0.001}
# model= lgb.train(params, train_data, 100)
# y_pred=model.predict(x_test)
# print(1-rmsle(y_test,y_pred))


# In[98]:


# from sklearn.ensemble import AdaBoostRegressor
# model = AdaBoostRegressor(n_estimators=1)
# model.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# print(1-rmsle(y_test,y_pred))


# In[99]:


# xg_predict=model.predict(df_test)


# In[100]:


# xg_predict


# In[95]:


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

