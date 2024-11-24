#!/usr/bin/env python
# coding: utf-8

# In[20]:


## Importing Libraries


# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


## Extracting the Data


# In[23]:


comments = pd.read_csv(r'C:\Users\stale\DS Portfolio Projects/UScomments.csv', on_bad_lines='skip')


# In[24]:


comments.head()


# In[25]:


## Getting Rid of Null Values


# In[26]:


comments.isnull().sum()


# In[27]:


comments.dropna(inplace=True)


# In[28]:


comments.isnull().sum()


# In[29]:


## Sentiment Analysis


# In[30]:


get_ipython().system('pip install textblob')


# In[31]:


from textblob import TextBlob


# In[32]:


comments.head(6)


# In[33]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[ ]:





# In[34]:


comments.shape


# In[35]:


sample_df = comments[0:1000] # Using sample data because of how big the data is. #


# In[36]:


sample_df.shape


# In[121]:


polarity = []

for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[123]:


len(polarity)


# In[125]:


# Polarity doesn't match the number of rows in the 'comments' DataFrame. 
# Which prevents Pandas from assigning the 'polarity' values to the 'comments' DataFrame.


# In[127]:


print(f"Length of comments: {len(comments['comment_text'])}")
print(f"Length of polarity: {len(polarity)}")


# In[ ]:





# In[130]:


## Resetting the DataDrame Index and Trimming the polarity to match the DataFrame length.


# In[132]:


comments = comments.reset_index(drop=True) # Reset DataFrame Index


# In[134]:


polarity = polarity[:len(comments)] # Trim polarity to match DataFrame Length


# In[136]:


print(f"Length of comments: {len(comments['comment_text'])}")
print(f"Length of polarity: {len(polarity)}")


# In[ ]:





# In[139]:


comments['polarity'] = polarity


# In[141]:


comments.head(5)


# In[ ]:





# In[144]:


# Wordcloud Analysis


# In[ ]:





# In[147]:


filter1 = comments['polarity']==1


# In[149]:


comments_positive = comments[filter1]


# In[151]:


comments_positive.head(5)


# In[ ]:





# In[154]:


filter2 = comments['polarity']==-1


# In[156]:


comments_negative = comments[filter2]


# In[158]:


comments_negative.head(5)


# In[ ]:





# In[161]:


# Installing Wordcloud


# In[163]:


get_ipython().system('pip install wordcloud')


# In[165]:


from wordcloud import WordCloud , STOPWORDS


# In[167]:


set(STOPWORDS)


# In[169]:


comments['comment_text']


# In[171]:


type(comments['comment_text'])


# In[173]:


total_comments_positive = ' '.join(comments_positive['comment_text'])


# In[175]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


# In[177]:


plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:





# In[180]:


total_comments_negative = ' '.join(comments_negative['comment_text'])


# In[182]:


wordcloud2 = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[184]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[ ]:





# In[ ]:





# In[188]:


# Perfeorming Emoji Analysis


# In[ ]:





# In[191]:


get_ipython().system('pip install emoji')


# In[192]:


import emoji


# In[193]:


comments['comment_text'].head(6)


# In[197]:


comment = 'trending 😉'


# In[199]:


[char for char in comment if char in emoji.EMOJI_DATA]


# In[205]:


emoji_list = []
for char in comment:
    if char in emoji.EMOJI_DATA:
        emoji_list.append(char)


# In[207]:


emoji_list


# In[ ]:





# In[ ]:





# In[243]:


all_emojis_list = []

for comment in comments['comment_text'].dropna():
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)


# In[244]:


all_emojis_list[0:10]


# In[ ]:





# In[250]:


from collections import Counter


# In[252]:


Counter(all_emojis_list).most_common(10)


# In[ ]:





# In[255]:


Counter(all_emojis_list).most_common(10)[0]


# In[259]:


Counter(all_emojis_list).most_common(10)[0][0]


# In[265]:


Counter(all_emojis_list).most_common(10)[1][0]


# In[267]:


Counter(all_emojis_list).most_common(10)[0][1]


# In[ ]:





# In[ ]:


# Defining Emojis


# In[271]:


emojis = [Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]


# In[ ]:





# In[ ]:


# Defining the frequency of emojis


# In[273]:


freqs = [Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]


# In[275]:


freqs


# In[ ]:





# In[290]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[292]:


trace = go.Bar(x=emojis , y=freqs)


# In[ ]:


# Top 5 emojis


# In[296]:


iplot([trace])


# In[298]:


# Collecting Youtube Data 


# In[ ]:





# In[300]:


import os


# In[312]:


files = os.listdir(r'C:\Users\stale\DS Portfolio Projects\additional_data')


# In[315]:


files


# In[ ]:





# In[320]:


files_csv = [file for file in files if '.csv' in file]


# In[322]:


files_csv


# In[ ]:





# In[324]:


files_json = [file for file in files if '.json' in file]


# In[331]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:





# In[340]:


full_df = pd.DataFrame()
path = r'C:\Users\stale\DS Portfolio Projects\additional_data'


# Listing all csv files in the directory
for file in files_csv:
    current_df = pd.read_csv(os.path.join(path, file), encoding='iso-8859-1', on_bad_lines='skip')

    full_df = pd.concat([full_df , current_df] , ignore_index=True)


# In[342]:


full_df.shape


# In[ ]:





# In[345]:


# Exporting data into (csv, json, db)


# In[ ]:





# In[354]:


full_df[full_df.duplicated()].shape #Shows that there is more than 36,000 rows duplicated


# In[ ]:





# In[357]:


full_df = full_df.drop_duplicates()


# In[359]:


full_df.shape #Final number of rows without duplicates


# In[ ]:





# In[364]:


full_df[0:1000].to_csv(r'C:\Users\stale\DS Portfolio Projects\YouTube Case-Study/youtube_sample.csv' , index=False)


# In[368]:


full_df[0:1000].to_json(r'C:\Users\stale\DS Portfolio Projects\YouTube Case-Study/youtube_sample.json')


# In[ ]:





# In[371]:


from sqlalchemy import create_engine


# In[378]:


engine = create_engine(r'sqlite:///C:\Users\stale\DS Portfolio Projects\YouTube Case-Study/youtube_sample.sqlite')


# In[ ]:





# In[381]:


full_df[0:1000].to_sql('Users' , con=engine , if_exists='append')


# In[ ]:





# In[384]:


# Finding out which category has the most likes


# In[ ]:





# In[387]:


full_df.head(5) #There's only category ids, but a category name is needed. 


# In[ ]:





# In[390]:


full_df['category_id'].unique()


# In[ ]:





# In[397]:


json_df = pd.read_json(r'C:\Users\stale\DS Portfolio Projects\additional_data/US_category_id.json')


# In[399]:


json_df


# In[403]:


json_df['items'][0]


# In[ ]:





# In[410]:


cat_dict = {}

for item in json_df['items'].values:
    cat_dict[int(item['id'])] = item['snippet']['title']


# In[412]:


cat_dict


# In[415]:


full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[419]:


full_df.head(4) #There os now a Category Name column


# In[ ]:





# In[432]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name' , y='likes' , data=full_df)
plt.xticks(rotation='vertical')

# As you can see, the most liked category is Music. 


# In[ ]:





# In[435]:


# Finding out whether the audience is engaged or not


# In[ ]:





# In[438]:


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[440]:


full_df.columns


# In[442]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name' , y='like_rate' , data=full_df)
plt.xticks(rotation='vertical')
plt.show()


# In[444]:


sns.regplot(x='views' , y='likes' , data = full_df)


# In[ ]:





# In[447]:


full_df.columns


# In[455]:


full_df[['views' , 'likes' , 'dislikes']].corr()


# In[459]:


sns.heatmap(full_df[['views' , 'likes' , 'dislikes']].corr() , annot=True)


# In[ ]:





# In[462]:


# Analyzing which channels have the largest number of trending videos


# In[ ]:





# In[465]:


full_df.head(6)


# In[ ]:





# In[468]:


full_df['channel_title'].value_counts()


# In[477]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[515]:


cdf = cdf.rename(columns={0:'total_videos'})


# In[ ]:





# In[509]:


import plotly.express as px


# In[517]:


px.bar(data_frame=cdf[0:20] , x='channel_title' , y='total_videos')


# In[ ]:




