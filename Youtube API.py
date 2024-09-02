from googleapiclient.discovery import build
import pandas as pd
from IPython.display import JSON

# Data Visuals Packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from dateutil import parser

#NLP (Natural Language Processing)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud

api_key = 'xxxxx'

channel_ids = ['UCJQJAI7IjbLcpsjWdSzYz0Q' , 
              # you can add more channels here, 
              ] 

api_service_name = "youtube"
api_version = "v3"
   
# Get credentials and create an API client
youtube = build(
    api_service_name, api_version, developerKey=api_key)

def get_channel_stats(youtube, channel_ids):

 # Get channel stats ↓

    all_data = []

    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)    
        )
    response = request.execute()

    #loop through items

    for item in response['items']:
        data = {'channelName': item['snippet']['title'],
               'subscribers': item['statistics']['subscriberCount'],
                'views': item['statistics']['viewCount'],
               'totalViews': item['statistics']['videoCount'],
               'playlistId': item['contentDetails']['relatedPlaylists']['uploads']
               }
        all_data.append(data)
        return(pd.DataFrame(all_data))

channel_stats = get_channel_stats(youtube, channel_ids)

channel_stats

playlist_id = 'UUJQJAI7IjbLcpsjWdSzYz0Q'

#Get video Ids
video_ids = get_video_ids(youtube, playlist_id)
len(video_ids)

#Get video details
video_df = get_video_details(youtube, video_ids)
video_df
# Get video comments

def get_comments_in_videos(youtube, video_ids):
    all_comments = []

    for video_id in video_ids:
        request = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id
        )
        response = request.execute()

        comments_in_video = [
            comment['snippet']['topLevelComment']['snippet']['textOriginal']
            for comment in response['items']
        ]
        comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}

        all_comments.append(comments_in_video_info)

    return pd.DataFrame(all_comments)

comments_df = get_comments_in_videos(youtube, video_ids)
comments_df ['comments'] [0]

#Check for NULL values
video_df.isnull().any()

#Check data types
video_df.dtypes

#Convert columns to numeric
numeric_cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)

#Publish day in the week
video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x: parser.parse(x))
video_df['publishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A"))

#Convert duration to seconds 
import isodate
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')

#Add tag count
video_df['tagCount'] = video_df['tags'].apply(lambda x: 0 if x is None else  len(x))
video_df

## EDA (Exploratory Data Analysis)

#Best performing videos 
sorted_video_df = video_df.sort_values('viewCount', ascending=False).iloc[0:9]
ax = sns.barplot(x='title', y='viewCount', data=sorted_video_df)
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.yaxis.set_major_formatter(ticker.FunctionFormatter(lambda x, pos: '{,.0f}'.format(x/1000) + 'K'))

#Worst perofming videos 
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=True)[0:9])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000) + 'K'))

##Video distribution per video 
fig, ax = plt.subplots(1, 2)
sns.scatterplot(data = video_df, x = 'commentCount', y = 'viewCount', ax = ax[0])
sns.scatterplot(data = video_df, x = 'likeCount', y = 'viewCount', ax = ax[1])

##Video duration

sns.histplot(data = video_df, x = 'durationSecs', bins=30)

##Wordcloud for video titles

stop_words = set(stopwords.words('english'))
video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item.lower() not in stop_words])

all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words)

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud)
    plt.axis("off");

wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black',
                     colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)

#Upload Schedule
day_df = pd.DataFrame(video_df['publishDayName'].value_counts())
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_df = day_df.reindex(weekdays)
day_df = day_df.reset_index()
day_df.columns = ['Day', 'Count']
ax = day_df.plot.bar(x='Day', y='Count', rot=0)


