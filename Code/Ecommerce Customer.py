#!/usr/bin/env python
# coding: utf-8

# In[307]:


#####################################################################################################
######################### ECOMMERCE DATA SET  #####################################################
#####################################################################################################


# In[308]:


##########################################################################
############### Part I - Importing 
##########################################################################


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[309]:


df = pd.read_csv('Ecommerce Customers.csv')


# In[310]:


df.head()


# In[311]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[312]:


df[df.duplicated()]                                #### no duplicates


# In[313]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[314]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### seems like a clean data


# In[315]:


df.isnull().any()


# In[316]:


df.info()


# In[317]:


######################################################################
############## Part IV - Feature Engineering
######################################################################


# In[318]:


df.head()


# In[319]:


x = df.Email[0]

x


# In[320]:


x.split('@')[1]


# In[321]:


df['Domain'] = df.Email.apply(lambda x:x.split('@')[1])     


# In[322]:


df.head()                      #### made a new column Domain to just see the domain


# In[323]:


df.Domain.value_counts()                #### most popular one is gmail and hotmail followed by yahoo


# In[324]:


df['Cities'] = df.Address.apply(lambda x:x.split()[-2])


# In[325]:


df.rename(columns={'Cities':'State'},inplace=True)


# In[326]:


df.head()                                 #### now we have all the states


# In[327]:


df.State.value_counts()                    #### now we have the states of customers


# In[328]:


new_df = df.copy()


# In[329]:


new_df.head()                        #### its always better and safer to work on the copy then the real df


# In[330]:


new_df.drop(columns=['Email','Address','Avatar'],inplace=True)           #### dropping two columns because we have stripped the important thing from it


# In[331]:


new_df.head()


# In[332]:


new_df.rename(columns={'Avg. Session Length':'Avg_session_Len',
                        'Time on App':'App_time',
                        'Time on Website':'Website_time',
                        'Length of Membership':'Membership_len',
                        'Yearly Amount Spent':'Spent_anually'},inplace=True)


# In[333]:


new_df.head()                  #### renamed the columns for easy calls


# In[334]:


new_df.Membership_len.round()


# In[335]:


new_df.head()


# In[336]:


new_df['Member'] = new_df.Membership_len.round()              #### keeping membership float doesn't makes sense here so for now we make a new col to round it off


# In[337]:


new_df.head()


# In[338]:


######################################################################
############## Part V - EDA
######################################################################


# In[339]:


new_df['Avg_session_Len'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Ecommerce Time Spent Graph')

plt.xlabel('Number of customers')

plt.ylabel('Time')


#### seems like the avg is around 33 


# In[340]:


new_df.Avg_session_Len.mean()


# In[341]:


new_df.Avg_session_Len.std()                #### seems pretty good nothing suspecious


# In[342]:


new_df['App_time'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='purple',color='black')

plt.title('Ecommerce App Time Graph')

plt.xlabel('Number of customers')

plt.ylabel('Time_App')


#### mean of the time on app is around 12


# In[343]:


new_df['Website_time'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black')

plt.title('Ecommerce Website Time Graph')

plt.xlabel('Number of customers')

plt.ylabel('Time_Website')


#### seems like from this info most of the people prefer the website then compared to app


# In[344]:


new_df.Website_time.mean()


# In[345]:


new_df['Spent_anually'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='grey')

plt.title('Ecommerce Spent Graph')

plt.xlabel('Number of customers')

plt.ylabel('Spent_anually')


#### seems like the mean is around 500 spent annually


# In[346]:


new_df.Spent_anually.mean()              #### we were right about it


# In[347]:


new_df.Spent_anually.std()               #### std seems a bit more on the higher side


# In[348]:


custom = {0:'black',
         1:'red',
         2:'pink',
         3:'grey',
         4:'orange',
         5:'green',
         6:'purple',
         7:'blue'}

g = sns.jointplot(x=new_df.Website_time,y=new_df.Spent_anually,data=new_df,hue='Member',palette=custom)

g.fig.set_size_inches(17,9)


#### clearly we see a peak in website time usage from members 3-4 years
#### also we see a peak in money spent from members of 3 and 4 years, interesting


# In[349]:


new_df.Member.unique()


# In[350]:


new_df.Member.value_counts()                #### seems like we have the most from 4 and 3 years plus members so it makes sense the time and spending is higher from those group


# In[351]:


new_df.head()


# In[352]:


g = sns.jointplot(x='App_time',y='Spent_anually',data=new_df,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)


#### seems pretty linear to me which obviously is not suprising


# In[353]:


from scipy.stats import pearsonr


# In[354]:


co_eff, p_value = pearsonr(new_df.App_time,new_df.Spent_anually)


# In[355]:


co_eff                                    #### we were right they are strongly correlated


# In[356]:


p_value                                   #### we can accept the alternative hypothesis


# In[357]:


corr = new_df.corr()


# In[358]:


corr


# In[359]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### something strange we see from here, website time is not correlated to money spend


# In[360]:


g = sns.jointplot(x='Website_time',y='Spent_anually',data=new_df,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)


#### no wonder this is not correlated


# In[361]:


co_eff, p_value = pearsonr(new_df.Website_time,new_df.Spent_anually)


# In[362]:


co_eff                         #### very suprised because initially i had thought they must be correlated but here we see its not the case


# In[363]:


p_value                        #### null hypothesis accepted


# In[364]:


state_df = new_df.groupby('State').sum()

state_df.head()


# In[365]:


from sklearn.preprocessing import StandardScaler


# In[366]:


scaler = StandardScaler()                          #### we will standardize state df


# In[367]:


standardized_df = scaler.fit_transform(state_df)


# In[368]:


df_comp = pd.DataFrame(standardized_df,columns=['Avg_session_Len', 'App_time', 'Website_time', 'Membership_len','Spent_anually', 'Member'])


# In[369]:


df_comp.head()


# In[370]:


df_comp.index = state_df.index                    #### we want the index from state df


# In[371]:


df_comp.head()


# In[372]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(df_comp,linewidths=0.1,ax=ax,cmap='viridis')


#### seems like state AA and AE are dominating here all across the board


# In[373]:


new_df[new_df.State == 'AA']                 #### we were right for some reason people from this state are our best customers


# In[374]:


new_df.State.value_counts()           #### the majority are from that state


# In[375]:


heat = df_comp.sort_values(by='Spent_anually',ascending=False).head(10)          #### top 10 by money spent states

heat


# In[376]:


fig, ax = plt.subplots(figsize=(25,15)) 

sns.heatmap(heat,annot=True,linewidths=0.5,ax=ax,cmap='viridis')


#### from this we can deduce to spend more resources to these states as they tend to be our best customers
#### lets see our worst customer loyalty group so we can do more campaign to bring them to our site


# In[377]:


df_comp.sort_values(by='Spent_anually',ascending=True).head(10)

#### clearly we can see that we need to do some work on these states to gravitate their population to our product and site


# In[378]:


df_comp.sort_values(by='Spent_anually')['Spent_anually'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=10,linestyle='dashed',linewidth=3,color='red')


#### note this is from standardization, hence we see negative on y axis


# In[379]:


new_df[new_df.State=='AA']['Member'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=15,linestyle='dashed',linewidth=4,color='red')


#### because AA state is so strongly spending on our site we wanted to see their membership status, seems like most of them are 3+ years


# In[380]:


sns.catplot(x='Member',data=new_df,kind='count',height=7,aspect=2,palette=custom)


#### seems like 3-4 years members are the sweat spot for us


# In[381]:


pl = sns.FacetGrid(new_df,hue='Member',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Spent_anually',fill=True)

pl.set(xlim=(0,new_df.Spent_anually.max()))

pl.add_legend()


#### lets take care of this error first


# In[382]:


variance_check = new_df.groupby('Member')['Spent_anually'].var()


# In[383]:


variance_check


# In[384]:


valid_members = variance_check[variance_check > 0].index
filtered_df = new_df[new_df['Member'].isin(valid_members)]


# In[385]:


pl = sns.FacetGrid(filtered_df,hue='Member',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Spent_anually',fill=True)

pl.set(xlim=(0,new_df.Spent_anually.max()))

pl.add_legend()


# In[386]:


new_df[new_df.Member==7]                #### been a member since 7 years


# In[387]:


new_df[new_df.Member==0]                #### new member but kde will have a hard time plotting this one so we took care with variance


# In[388]:


filtered_df.Member.unique()


# In[389]:


pl = sns.FacetGrid(filtered_df,hue='Member',aspect=4,height=4)

pl.map(sns.kdeplot,'App_time',fill=True)

pl.set(xlim=(0,filtered_df.App_time.max()))

pl.add_legend()


#### seems like the peak is around 12 for all members except for 6 years members


# In[390]:


filtered_df.head()


# In[391]:


filtered_df.State.unique()


# In[392]:


filtered_df['State_num'] = filtered_df.State.map({'MI':1,
                                                  'CA':2,
                                                  'DC':3,
                                                  'OH':4,
                                                  'PR':5,
                                                  'MN':6,
                                                  'WV':7,
                                                  'AP':8,
                                                  'SD':9,
                                                  'AA':10,
                                                  'WY':11,
                                                  'MO':12,
                                                  'MP':13,
                                                  'ND':14,
                                                  'GA':15,
                                                  'PW':16,
                                                  'MT':17,
                                                  'KY':18,
                                                  'AE':19,
                                                  'VI':20,
                                                  'TX':21,
                                                  'MS':22,
                                                  'SC':23,
                                                  'WA':24,
                                                  'NJ':25,
                                                  'NH':26,
                                                  'ME':27,
                                                  'ID':28,
                                                  'TN':29,
                                                  'AK':30,
                                                  'DE':31,
                                                  'FM':32,
                                                  'HI':33,
                                                  'KS':34,
                                                  'NC':35,
                                                  'UT':36,
                                                  'AL':37,
                                                  'LA':39,
                                                  'NE':40,
                                                  'OR':41,
                                                  'CT':42,
                                                  'MA':43,
                                                  'IN':44,
                                                  'AZ':45,
                                                  'MH':46,
                                                  'NY':47,
                                                  'CO':48,
                                                  'IA':49,
                                                  'GU':50,
                                                  'AS':51,
                                                  'RI':52,
                                                  'VA':53,
                                                  'MD':54,
                                                  'OK':55,
                                                  'WI':56,
                                                  'VT':57,
                                                  'FL':58,
                                                  'IL':59,
                                                  'NV':60,
                                                  'PA':61,
                                                  'NM':62,
                                                  'AR':63})

#### dont worry about the warning, we should have gone with a copy to prevent this but at this point lets just continue


# In[393]:


filtered_df.head()


# In[394]:


pl = sns.catplot(y='State_num',x='Member',data=filtered_df,kind='point',height=10,aspect=2,color='black')


#### this is very interesting plot, from here we see that most of the members are 1 year long member and 6 years long members from different states


# In[395]:


pl = sns.lmplot(x='Spent_anually',y='Avg_session_Len',data=new_df,hue='Member',height=7,aspect=2,palette=custom)


#### quite interesing plot, we can clearly see linear relationship between different memberships
#### the strongest ones being red and orange


# In[396]:


sns.lmplot(x='Spent_anually',y='Member',data=new_df,x_bins=[257,300,320,350,380,400,420,450,480,500,520,550,580,600,620,650,680,700,720,750,760],height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'})


#### this is just amazing, like a really good proper linear correlation. sorry I get excited seeing this strong relationship. we love stats here a lot


# In[397]:


sns.lmplot(x='Membership_len',y='Spent_anually',data=new_df,height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'})


#### just amazing amazing linear relationship between spending and membership length, just love seeing this honestly


# In[398]:


new_df.Spent_anually.std()


# In[399]:


new_df.Spent_anually.mean()


# In[400]:


mean_df = new_df.Spent_anually.mean()

std_df = new_df.Spent_anually.std()


# In[401]:


from scipy.stats import norm


# In[405]:



x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(20, 7))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')

#### areas under the curve
plt.fill_between(x, y, where=(x >= mean_df - std_df) & (x <= mean_df + std_df), color='green', alpha=0.2, label='68%')
plt.fill_between(x, y, where=(x >= mean_df - 2*std_df) & (x <= mean_df + 2*std_df), color='orange', alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mean_df - 3*std_df) & (x <= mean_df + 3*std_df), color='red', alpha=0.2, label='99.7%')

#### mean and standard deviations
plt.axvline(mean_df, color='black', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 3*std_df, color='yellow', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 3*std_df, color='yellow', linestyle='dashed', linewidth=1)

plt.text(mean_df, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, plt.gca().get_ylim()[1]*0.05, f'z=1    {mean_df + std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, plt.gca().get_ylim()[1]*0.05, f'z=-1   {mean_df - std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=2  {mean_df + 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-2 {mean_df - 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=3  {mean_df + 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-3 {mean_df - 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')


#### annotate the plot
plt.text(mean_df, max(y), 'Mean', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, max(y), '-1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, max(y), '+1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, max(y), '-2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, max(y), '+2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, max(y), '-3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, max(y), '+3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')

#### labels
plt.title('Spent Annually distribution inside the Ecommerce Dataset')
plt.xlabel('Spending')
plt.ylabel('Probability Density')

plt.legend()



#### note this is very infomative as we see the majority of spending is between 420-578 with the peak being 499
#### then we move to z-score 2 where the density decreases on both side on z scores and then same happens with z-score level 3 on both sides


# In[406]:


#######################################################################################
############## PART VI - Model - Linear Regression
#######################################################################################


# In[407]:


X = new_df.drop(columns=['Spent_anually','Membership_len'])

X.head()


# In[408]:


y = new_df['Spent_anually']

y.head()


# In[409]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[410]:


preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Domain','State']),
                                               ('num', StandardScaler(),['Avg_session_Len','App_time','Website_time','Member'])
                                              ]
                                )


# In[411]:


from sklearn.pipeline import Pipeline


# In[412]:


from sklearn.linear_model import LinearRegression


# In[413]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())
                       ])


# In[414]:


from sklearn.model_selection import train_test_split


# In[415]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[416]:


model.fit(X_train, y_train)


# In[417]:


y_predict = model.predict(X_test)


# In[418]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


#### seems like a perfect model from the plot, this is how it should be without any pattern formation


# In[419]:


from sklearn import metrics


# In[420]:


metrics.r2_score(y_test,y_predict)                #### amazing


# In[421]:


metrics.mean_squared_error(y_test,y_predict) 


# In[422]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))              #### seems like we are off by 23 


# In[423]:


from sklearn.model_selection import GridSearchCV                   #### lets see if we can further improve it


# In[424]:


from sklearn.ensemble import RandomForestRegressor


# In[425]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])


# In[426]:


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}


# In[427]:


get_ipython().run_cell_magic('time', '', "\ngrid_model = GridSearchCV(model, param_grid, cv=5, scoring='r2',verbose=2)\ngrid_model.fit(X_train, y_train)")


# In[428]:


best_model = grid_model.best_estimator_


# In[429]:


y_predict = best_model.predict(X_test)


# In[430]:


metrics.r2_score(y_test,y_predict)            #### seems some improvement


# In[431]:


metrics.mean_squared_error(y_test,y_predict)


# In[432]:


rmse = np.sqrt(metrics.mean_squared_error(y_test,y_predict))

rmse                          #### now we are off my 21.8


# In[433]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


#### better


# In[434]:


from xgboost import XGBRegressor


# In[435]:


from sklearn.model_selection import RandomizedSearchCV


# In[436]:


from scipy.stats import randint


# In[437]:


from scipy.stats import uniform


# In[438]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])


# In[439]:


param_grid = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(3, 10),
    'regressor__min_child_weight': randint(1, 10),
    'regressor__subsample': uniform(0.5, 0.5),
    'regressor__colsample_bytree': uniform(0.5, 0.5)
}


# In[440]:


random_model = RandomizedSearchCV(model, param_grid, cv=5, scoring='r2', n_iter=100, random_state=42,verbose=2)


# In[441]:


get_ipython().run_cell_magic('time', '', '\nrandom_model.fit(X_train, y_train)')


# In[442]:


best_model = random_model.best_estimator_


# In[443]:


y_predict = best_model.predict(X_test)


# In[444]:


metrics.mean_squared_error(y_test,y_predict)


# In[445]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))         #### again some improvement went from 21.8 to 19.9 


# In[446]:


metrics.r2_score(y_test,y_predict)


# In[447]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='grey')

plt.axhline(0,color = 'black',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### just beautiful and well fit model, usually this method is used to see if the model is better fit for the data set and yes it is


# In[448]:


plt.figure(figsize=(10,6))

plt.scatter(y_test,y_predict,color='black')

plt.xlabel('actual')

plt.ylabel('predicted')


#### see even without a linear line we can see a pattern and how well they are, this is just beauty


# In[449]:


#### we are going back to the training because I think we can improve by excluding 2 feature columns which is not highly correlated

new_df.head()


# In[450]:


X = new_df.drop(columns=['Spent_anually','Domain','State','Membership_len'])

X.head()


# In[451]:


y = new_df['Spent_anually']

y.head()


# In[452]:


preprocessor = ColumnTransformer(transformers=[
                                               ('num', StandardScaler(),['Avg_session_Len','App_time','Website_time','Member'])
                                              ]
                                )


# In[453]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())
                       ])


# In[454]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[455]:


model.fit(X_train, y_train)


# In[456]:


y_predict = model.predict(X_test)


# In[457]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


#### seems like a perfect model from the plot, this is how it should be


# In[458]:


plt.figure(figsize=(10,6))

plt.scatter(y_test,y_predict,color='black')

plt.xlabel('actual')

plt.ylabel('predicted')


#### good one


# In[459]:


metrics.r2_score(y_test,y_predict)


# In[460]:


metrics.mean_squared_error(y_test,y_predict)


# In[461]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))        #### see even with very basic linear model, we are surpassing what we did with advanced methods


# In[462]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])


# In[463]:


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}


# In[464]:


get_ipython().run_cell_magic('time', '', "\ngrid_model = GridSearchCV(model, param_grid, cv=5, scoring='r2')\ngrid_model.fit(X_train, y_train)")


# In[465]:


best_model = grid_model.best_estimator_


# In[466]:


y_predict = best_model.predict(X_test)


# In[467]:


metrics.r2_score(y_test,y_predict)


# In[468]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))         #### made it worse


# In[469]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])


# In[470]:


param_grid = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(3, 10),
    'regressor__min_child_weight': randint(1, 10),
    'regressor__subsample': uniform(0.5, 0.5),
    'regressor__colsample_bytree': uniform(0.5, 0.5)
}


# In[471]:


random_model = RandomizedSearchCV(model, param_grid, cv=5, scoring='r2', n_iter=100, random_state=42)


# In[472]:


get_ipython().run_cell_magic('time', '', '\nrandom_model.fit(X_train, y_train)')


# In[473]:


best_model = random_model.best_estimator_


# In[474]:


y_predict = best_model.predict(X_test)


# In[475]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))


# In[476]:


metrics.r2_score(y_test,y_predict)                     #### not the improvement we were hoping for but its alright


# In[477]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='green')

plt.axhline(0,color = 'black',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[478]:


plt.figure(figsize=(10,6))

plt.scatter(y_test,y_predict,color='green')

plt.xlabel('actual')

plt.ylabel('predicted')


# In[479]:


################################################################################################################
#### We conducted an extensive analysis of an Ecommerce dataset to predict annual customer spending. ###########
#### Through thorough Exploratory Data Analysis (EDA) and Feature Engineering (FE), we developed various #######
#### regression models. Our best-performing model achieved a Root Mean Squared Error (RMSE) of 19.9 and a ######
#### coefficient of determination (R²) of 0.92. Additionally, a Random Forest Regression model yielded an ######
#### RMSE of 21.8, showcasing its robustness as a predictive tool. #############################################
################################################################################################################

