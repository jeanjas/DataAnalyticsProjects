#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'C:\Users\stale\DS Portfolio Projects\Video Game Sales Analysis/vgsales.csv')


# In[3]:


print(df.head())


# In[4]:


print(df.dtypes)


# In[5]:


# Replace NaN values with the median year
df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)

df['Year'] = df['Year'].astype(int) #Converting the Year into Integers


# In[6]:


print(df.head())


# In[7]:


df.isnull().sum() #Checking for null values


# In[8]:


df['Publisher'] = df['Publisher'].fillna('Unknown') #Filling 'Publisher' null values with 'Unknown'


# In[9]:


df.isnull().sum() #No more null values, yay! :) 


# In[10]:


print(df.duplicated().sum) #Checking for duplicates


# In[ ]:





# In[11]:


#Top perforoming genres and platforms globally

genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)

print(genre_sales, platform_sales)


# In[ ]:





# In[12]:


#Sales distribution by Region

regional_sales = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

regional_sales.plot(kind='bar', title='Sales by Region')

plt.show()


# In[ ]:





# In[13]:


#Sales by Year Analysis 

sales_by_year = df.groupby('Year')['Global_Sales'].sum()
sales_by_year.plot(kind='line', title='Global Sales Over Time')

plt.show()


# In[ ]:





# In[14]:


#Top 10 Publishers 

top_publishers = df.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10)

print(top_publishers)


# In[15]:


# Most Popular Generes in Different Regions

genre_by_region = df.groupby(['Genre'])[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

print(genre_by_region)


# In[16]:


genre_by_region.shape


# In[17]:


from matplotlib_inline.backend_inline import set_matplotlib_formats

# Set higher resolution for inline plots
set_matplotlib_formats('retina')


# In[18]:


# Analysizing Platform Sales Overtime


platform_sales_by_year = df.groupby(['Year', 'Platform'])['Global_Sales'].sum().unstack()
platform_sales_by_year.plot(kind='line', title='Platform Sales OVer Time', figsize=(20,10))

plt.show()


# In[19]:


# Select top platforms by total sales
top_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(5).index
filtered_data = df[df['Platform'].isin(top_platforms)]

#Re-plot with limited platforms
platform_sales_by_year = filtered_data.groupby(['Year', 'Platform'])['Global_Sales'].sum().unstack()
platform_sales_by_year.plot(kind='line', title='Platform Sales Over Time', figsize=(12, 6))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[20]:


# Predictive Modeling


# In[ ]:





# In[21]:


#Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split

x=df[['Name', 'Platform', 'Genre']]
y=df['Global_Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[22]:


print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Optional: Display the first few rows of each dataset
print("\nx_train sample:\n", x_train.head())
print("\ny_train sample:\n", y_train.head())


# In[ ]:




