#!/usr/bin/env python
# coding: utf-8

# In[384]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("/Users/nikitacarolyn/Downloads/airbnb.csv")


# In[385]:


df.head()


# In[386]:


df.columns


# In[388]:


df.shape


# In[273]:


df.columns=(
     df.columns.str.lower()
    .str.replace(' ', '_')
    .str.replace('restraunt','restaurant')
)
df.head()


# In[274]:


bool_cols = ['shared_room', 'private_room', 'superhost']
for col in bool_cols:
    df[col] = df[col].astype(int)


# In[275]:


df.nunique()


# In[276]:


df.info()


# In[277]:


df.isnull().sum()


# In[278]:


df.duplicated().sum()


# In[279]:


df.describe().T


# ## Graph showing Price wrt City

# In[280]:


sns.histplot(data = df, x = 'city', bins = 20, alpha=0.3, hue='city', legend = False)
plt.title('Airbnb Price by City', fontsize=13, weight='bold')
plt.xticks(rotation=45)
plt.xlabel('City')
plt.ylabel('Count of city')


# # DATA TRANSFORMATIONS:

# In[281]:


numeric_cols = ['price', 'city_center_(km)', 'metro_distance_(km)', 'normalised_attraction_index', 
                'normalised_restaurant_index', 'cleanliness_rating', 'guest_satisfaction']

plt.figure(figsize=(14,6))

for i, column in enumerate(numeric_cols):
    plt.subplot(2,4,i+1)
    sns.histplot(data=df, x=column)
    col_title = column.replace('_', ' ').title()
    plt.title(col_title)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()


# From the histograms, there is primarily right-skewed distributions, however, left-skewed distribution for cleanliness and guest satisfaction. 
# Now, we have to scale these values using the min-max scale to maintain uniformity in the range of values. This also means normalized_attraction_index and normalized_restaurant_index are likely better predictors since they are closest to a normal distribution. 

# ## Outliers for price:

# In[382]:


sns.boxplot(df['price'])


# In[283]:


cap_5000 = df[df['price'] <= 5000]
cap_2500 = df[df['price'] <= 2500]
cap_1000 = df[df['price'] <= 1000]

rows_cut = [len(df) - len(airbnb) for airbnb in [cap_5000, cap_2500, cap_1000]]


# In[284]:


rows_cut


# 2500$ cap makes the most sense for generalisation since most of the highly expensive listings are gone, but 
# less than 50 listings are removed. 
# 476 is still quite small at roughly 1% of the overall data, however it is better not to eliminate too many data 
# points unless really required

# In[285]:


sns.histplot(df['price']);


# Now, the price column has a similar shape as some of the other features

# In[286]:


for col in numeric_cols:
    df['log_'+col] = df[col].apply(lambda x: np.log(x+1))
log_df = df[[col for col in df.columns if col.startswith('log_')]]
log_df.head()


# In[287]:


plt.figure(figsize=(16,8))

for i, column in enumerate(log_df.columns):
    plt.subplot(2,4,i+1)
    sns.histplot(data=log_df, x=column)
    col_title = column.replace('_', ' ').title()
    plt.title(col_title)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()


# Most of these distributions look more normally distributed and help meet assumptions of Pearson Correlation Coefficient.
# The log transform work better than the provided normalised form of Restrant and Attraction Index.
# However, cleanliness and satisfaction distributions are closer to discrete values than they are continuous

# # To mainly check the distribution of the target variable - Price

# In[288]:


sns.displot(data=df, x='price', hue='city', kind='kde', fill=True, palette=sns.color_palette('bright')[:9], height=5, aspect=2)


# From the graph, the price is not normally distributed. Some accomodations have a rather high price, hence we cannot talk about normal distribution at all. Therefore, we have to scale this value

# Since Price is our target variable, mainly focusing on Price, we see that it not normally distributed and its descriptive statistics are comparitively higher than the other variables and hence, we scale it.

# In[289]:


df.log_price.describe()


# After applying log10 to normalise the Price variable. Now we can prepare 3 plots using this to check how the price depends on city, day of the week and person capacity

# ## Distribution of Price wrt City, Day and Person Capacity

# In[290]:


sns.displot(data=df, x='log_price', hue='city', kind='kde', fill=True, palette=sns.color_palette('bright')[:9], 
            height=8, aspect=1.5).set(title='Price by cities')

sns.displot(data=df, x='log_price', hue='day', kind='kde', fill=True, palette=sns.color_palette('bright')[:9], 
            height=8, aspect=1.5).set(title='Price by days')

sns.displot(data=df, x='log_price', hue='person_capacity', kind='kde', fill=True, palette=sns.color_palette('bright')[:9], 
            height=8, aspect=1.5).set(title='Price by capacity')


# Conclusions from the graphs:
# 
# Price vs City - The most of the data points for accomodation is represented by Rome. Airbnb accommodation is chaper in Athenes, while Amsterdam and Paris are more expensive than the other Cities
# 
# Price wrt Days - From the graph, we see that the price of Airbnbs does not depend on the day of the week. On both, week days and weekends, the prices are mostly similar with no difference in which day is better to rent
# 
# Price vs Person capacity - We can conclude at least 2 things:
# -In the dataset, the most common person capacity offered by Airbnbs is 2
# -Prices for all the different person capacities are almost similar to each other. The only visible difference is that of person capacity 4 which is second most common and also little more expensive than the others, however it is not that significant.

#  Visitors leave the best satisfaction rating if the house was cleaned perfectly.

# We see that satisfaction of guests does not depend on the distance to the city centre. This means that the guests book accomodations mosrtly irrespective of this variable even if the accommodation is far away from the city centre but keeping the other variables (like cleanliness, price, room type etc) as priority.

# ## Dependence of Guest Satisfaction on Price

# In[292]:


sns.scatterplot(x='guest_satisfaction', y=df_log['Price'], data=df, legend='full').set(title='Dependence of Guest Satisfaction on Price');


# The guest satisfaction is comparitively high (from 70 to 100) when the price is moderate, that is neither too high, nor too low (2, 2.5 and 3. It may even go up to 3.5 but comaritively less). This could mean that customers are willing to pay a moderate price for a good enough accommodation. Cheaper accommodations might not have all the required facilities that a customer needs and an expensive accommodation may be way over budget for the customer.

# In[293]:


sns.boxplot(data=df_log, x='cut_labels_satisfaction', y=df_log['Price'], palette='Set3', saturation=1, width=0.7, whis=5,
            medianprops={"color": "black"}, flierprops={"marker": "x"}).set(title='Boxplot of Guest satisfaction and Price')


# Here, we see that the guest satisfaction does not depend on the price of the rooms. 

# ## Distance from City Centre and Price

# In[294]:


sns.boxplot(data=df_log, x='cut_labels_city_center', y=df_log['Price'], palette='Set3', saturation=1, width=0.7, whis=5,
            medianprops={"color": "black"}, flierprops={"marker": "x"}).set(title='Distance from City Centre and Price')


# The price is quite low wrt the distance and remains more or less constant from 0km to 12km

# In[342]:


plt.figure(figsize = (10, 7))
sns.histplot(data = df, x = 'log_price', hue = 'city', bins = 20, multiple = 'stack', kde = True)

plt.title('Price by City')
plt.xlabel('Price')
plt.ylabel('Count')


# In[343]:


plt.figure(figsize = (10, 7))
sns.boxplot(x = 'cleanliness_rating', y = 'log_price', data=df)

plt.title('Price by Cleanliness Rating')
plt.ylabel('Price')
plt.xlabel('Cleanliness Rating')


# In[344]:


plt.figure(figsize = (10, 7))
sns.boxplot(x = 'bedrooms', y = 'log_price', data=df)

plt.title('Price by number of Bedrooms')
plt.ylabel('Price')
plt.xlabel('Bedrooms')


# In[403]:


corrMat = df.corr(method='pearson')
sns.set(font_scale=0.8)
plt.figure(figsize=(16,10))
sns.heatmap(corrMat, cmap='RdBu_r',annot=True, vmin=-1, vmax=1,linewidths=0.5, linecolor='black',square=False,); 


# ## Machine Learning (Target - Price):

# In[364]:


list_str = df.select_dtypes(include = 'object').columns
le = LabelEncoder()

for c in list_str:
    df[c] = le.fit_transform(df[c])


# In[365]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X = df.drop(['price','shared_room','private_room',
             'superhost','attraction_index','restaurant_index', 'day',
             'cleanliness_rating'], axis = 1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[366]:


X=pd.get_dummies(X)
X.head()


# In[367]:


RF = RandomForestRegressor(n_estimators=1000)
RF.fit(X_train,y_train)

predict_RF = RF.predict(X_test)


# In[359]:


print('R2 Score: ', r2_score(y_test, predict_RF))
print("MSE: ", mean_squared_error(y_test, predict_RF))
print("MAE: ", mean_absolute_error(y_test, predict_RF))

results_df = pd.DataFrame(data=[["Random Forest Regression", *evaluate(y_test, predict_RF)]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])


# In[368]:


from sklearn.linear_model import LinearRegression


# In[369]:


LR = LinearRegression()
LR.fit(X_train,y_train)

LR_pred = LR.predict(X_test)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, LR_pred)


results_df_temp = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, LR_pred)]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])
results_df = results_df.append(results_df_temp, ignore_index=True)


# In[1]:


importances = df.feature_importances_


# In[401]:


import matplotlib.pyplot as plt
a, b = np.polyfit(y_test, predict_RF, 1)
plt.scatter(y_test, predict_RF)
plt.plot(y_test, a*y_test+b)   

plt.title("Goodness of fit  - Random Forest Regression")
plt.xlabel("y_test")
plt.ylabel("Predicted RF")
plt.grid(True)


# In[402]:


import matplotlib.pyplot as plt
a, b = np.polyfit(y_test, LR_pred, 1)
plt.scatter(y_test, LR_pred)
plt.plot(y_test, a*y_test+b)

plt.title("Goodness of fit  - Linear Regression")
plt.xlabel("y_test")
plt.ylabel("Predicted LR")
plt.grid(True)


# In[ ]:


plt.figure(figsize = (10,10))
fti.plot.barh();

