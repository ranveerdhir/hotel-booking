#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


df = pd.read_csv(r"C:\Users\ranveer\OneDrive - University of Victoria\Documents\ML project\hotel_bookings.csv")


# In[ ]:





# In[3]:


type(df)


# In[4]:


df.head(3)


# In[5]:


df.isnull()


# In[6]:


df.drop(['agent','company'],axis = 1,inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


df['country'].value_counts()


# In[8]:


df['country'].value_counts().index


# In[9]:


df['country'].fillna(df['country'].value_counts().index[0],inplace=True)


# In[10]:


df.fillna(0,inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


filter1 = (df['children'] == 0) &(df['adults']==0)&(df['babies']==0)


# In[13]:


df[filter1]


# In[14]:


data = df[~filter1]


# In[15]:


data.shape


# In[16]:


data['is_canceled'].unique()


# In[17]:


data[data['is_canceled']==0]['country'].value_counts()/75011


# In[ ]:





# In[18]:


len(data[data['is_canceled']==0])


# In[19]:


country_wise_data = data[data['is_canceled']==0]['country'].value_counts().reset_index()


# In[20]:


country_wise_data.columns = ['country','no_of_guests']


# In[21]:


country_wise_data


# In[22]:


get_ipython().system('pip install plotly')


# In[23]:


get_ipython().system('pip install chart_studio')


# In[24]:


import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode,plot, iplot
init_notebook_mode(connected=True)


# In[ ]:





# In[25]:


import plotly.express as px


# In[ ]:





# In[26]:


map_guest = px.choropleth(country_wise_data,locations = country_wise_data['country'],
             color=country_wise_data['no_of_guests'],
             hover_name=country_wise_data['country'],
             title = 'home country of guests')


# In[ ]:





# In[ ]:





# In[27]:


map_guest.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


#assuming payment is done in euro as all the popular countries are european 
data2 = data[data['is_canceled'] == 0 ]      #filter of for cancelation 


# In[29]:


data2.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


sns.boxplot(x ='reserved_room_type',y = 'adr',hue = 'hotel',data = data2)


# In[31]:


data_resort = data[(data['hotel']=='Resort Hotel') & (data['is_canceled']==0)]
data_city = data[(data['hotel']=='City Hotel') & (data['is_canceled']==0)]


# In[ ]:





# In[ ]:





# In[32]:


data_city


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


#which are the most busy month

#create 2 data frames , with respect to resort and with respet to city (hotel == resort and booking is not canceled)


# In[34]:


data['hotel'].unique() # returns unique values in the hotel feature


# In[35]:


rush_resort = data_resort['arrival_date_month'].value_counts().reset_index() #get frequecy table
rush_resort.columns = ['month','no_of_guests']  #setting column names
rush_resort


# In[36]:


rush_city = data_city['arrival_date_month'].value_counts().reset_index() #get frequecy table
rush_city.columns = ['month','no_of_guests']  #setting column names
rush_city


# In[37]:


# merging both data frames

final_rush = rush_resort.merge(rush_city,on = 'month')  #on tells the comman column


# In[ ]:





# In[38]:


final_rush.columns = ['month','no_of_guests_in_resort','no_of_guests_city']


# In[ ]:





# In[39]:


final_rush


# In[40]:


#!pip install sorted-months-weekdays


# In[ ]:





# In[41]:


#!pip install sort_dataframeby_monthorweek


# In[ ]:





# In[42]:


import sort_dataframeby_monthorweek as sd


# In[43]:


final_rush = sd.Sort_Dataframeby_Month(final_rush,'month') #sort it 


# In[44]:


final_rush.columns


# In[ ]:





# In[45]:


px.line(data_frame = final_rush,x = 'month',y = ['no_of_guests_in_resort','no_of_guests_city']) #august is most intense month of bookings


# In[46]:


#which month has the highest average daily rate


# In[47]:


data = sd.Sort_Dataframeby_Month(data,'arrival_date_month')


# In[ ]:





# In[48]:


sns.barplot(x = 'arrival_date_month', y = 'adr',data = data,hue  = 'is_canceled')
plt.xticks(rotation = 'vertical')
plt.show()


# In[49]:


# identify if bookings were made for weekends or weekdays or both 


# In[50]:


data.columns


# In[ ]:





# In[51]:


pd.crosstab(index=data['stays_in_weekend_nights'],columns =data['stays_in_week_nights'] )


# In[ ]:





# In[52]:


def week_function(row):
    feature1 = 'stays_in_weekend_nights'
    feature2 = 'stays_in_week_nights'
    
    if row[feature2] == 0 and row[feature1] > 0:
        return 'stay_just_weekend'
    
    elif row[feature2] > 0 and row[feature1] == 0:
        return 'stay_just_weekdays'
    elif row[feature2] > 0 and row[feature1] >0:
        return 'stay_both_weekdays_weekends'
    
    else:
        return 'undefined_data'
        
        


# In[53]:


data2['weekend_or_weekday'] = data2.apply(week_function,axis = 1)


# In[54]:


data2.head(2)


# In[55]:


data2['weekend_or_weekday'].value_counts()


# In[56]:


type(sd)


# In[57]:


data2 = sd.Sort_Dataframeby_Month(data2,'arrival_date_month')


# In[ ]:





# In[ ]:





# In[58]:


group_data = data2.groupby(['arrival_date_month','weekend_or_weekday']).size().unstack().reset_index() #called unstack to convert to matrix


# In[ ]:





# In[59]:


sorted_data = sd.Sort_Dataframeby_Month(group_data,'arrival_date_month')


# In[60]:


sorted_data.set_index('arrival_date_month',inplace = True)


# In[61]:


sorted_data


# In[62]:


sorted_data.plot(kind = 'bar',stacked = True)


# In[63]:


#create more suitable features so that ml modal can learn through this 


# In[64]:


data2.columns


# In[65]:


#can i utilize adult ,children and babies feature and create family
#for family to exist it must have adults and it can have children or babies || children and babies


# In[66]:


def family(row):
    if(row['adults'] > 0) & (row['children'] > 0 or row['babies'] > 0):
        return 1
    else:
        return 0


# In[67]:


data['is_family']=data.apply(family,axis = 1) #axis  is 1 because we are doing it on column basis


# In[68]:


data['total_customer'] = data['adults'] + data['babies'] + data['children']


# In[69]:


data['total_nights'] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']


# In[70]:


data.head(3)


# In[71]:


data['deposit_type'].unique()


# In[72]:


dict1 = {'No Deposit': 0,'Non Refund':0,'Refundable':0}


# In[73]:


data['deposit_given'] = data['deposit_type'].map(dict1)


# In[74]:


data.columns


# In[75]:


data.drop(columns=['adults', 'children', 'babies','deposit_type'],axis = 1,inplace =True)


# In[76]:


data.head(6)


# In[77]:


#apply feature encoding


# In[78]:


data.dtypes


# In[79]:


#hotel has categorical features


# In[80]:


#using loops and list comprehension for extraction 


# In[81]:


cate_features = [col for col in data.columns if data[col].dtype=='object']
    


# In[82]:


cate_features


# In[ ]:





# In[83]:


num_features = [col for col in data.columns if data[col].dtype!='object']


# In[ ]:





# In[84]:


num_features


# In[85]:


data_cat = data[cate_features]


# In[ ]:





# In[ ]:





# In[86]:


data.groupby(['hotel'])['is_canceled'].mean().to_dict()


# In[ ]:





# In[ ]:





# In[87]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


data_cat['cancellation'] = data['is_canceled']


# In[89]:


data_cat.head()


# In[90]:


cols = data_cat.columns


# In[91]:


cols


# In[92]:


cols = cols[0:-1]


# In[93]:


for col in cols:
       dict2 = data_cat.groupby([col])['cancellation'].mean().to_dict()
       data_cat[col] = data_cat[col].map(dict2)


# In[94]:


data_cat.head()


# In[95]:


data[num_features]


# In[96]:


dataframe = pd.concat([data_cat,data[num_features]],axis = 1)


# In[ ]:





# In[97]:


dataframe


# In[98]:


dataframe.drop(['cancellation'],axis = 1,inplace = True)


# In[99]:


dataframe.head(3)


# In[100]:


# need to handle outliers now 


# In[101]:


# do log tranformation to handle that 


# In[102]:


sns.displot(dataframe['lead_time'])


# In[ ]:





# In[103]:


def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[ ]:





# In[104]:


handle_outlier('lead_time')


# In[ ]:





# In[105]:


sns.displot(dataframe['lead_time'])


# In[106]:


## adr


# In[107]:


sns.displot(dataframe['adr'])


# In[108]:


dataframe[dataframe['adr'] < 0]


# In[ ]:





# In[109]:


handle_outlier('adr')


# In[110]:


sns.FacetGrid(data,hue = 'is_canceled',xlim = (0,500)).map(sns.kdeplot,'lead_time',shade = True).add_legend()


# In[111]:


corr = dataframe.corr()


# In[112]:


corr


# In[113]:


corr['is_canceled'].sort_values(ascending = False)


# In[114]:


corr['is_canceled'].sort_values(ascending = False).index


# In[ ]:





# In[115]:


features_to_drop=['reservation_status','reservation_status_date','arrival_date_year',
       'arrival_date_week_number', 'stays_in_weekend_nights',
       'arrival_date_day_of_month']


# In[ ]:





# In[116]:


dataframe.drop(features_to_drop,axis=1,inplace = True)


# In[117]:


dataframe.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[118]:


data


# In[119]:


#how to find important features for modal building


# In[120]:


dataframe.dropna(inplace = True)


# In[ ]:





# In[121]:


x = dataframe.drop('is_canceled',axis = 1)


# In[ ]:





# In[122]:


y = dataframe['is_canceled']


# In[ ]:





# In[123]:


#pass x and y to feature selection algorithm


# In[124]:


from sklearn.linear_model import Lasso


# In[125]:


from sklearn.feature_selection import SelectFromModel


# In[126]:


#Lasso(alpha = 0.005)


# In[127]:


feature_sel_model = SelectFromModel(Lasso(alpha = 0.005))


# In[128]:


feature_sel_model.fit(x,y)


# In[129]:


feature_sel_model.get_support()


# In[130]:


cols = x.columns


# In[131]:


selected_feature = cols[feature_sel_model.get_support()]


# In[132]:


selected_feature


# In[ ]:





# In[133]:


x = x[selected_feature]


# In[140]:


x


# In[ ]:


## prping test data


# In[135]:


from sklearn.model_selection import train_test_split


# In[136]:


X_train,X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)


# In[139]:


X_train


# In[141]:


y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




