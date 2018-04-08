
# Sentiment analysis of Sampled News Organization's Twitter Feeds

## Findings:

1. During the time period of the analysis, all sampled news organizations had a negative compound score
2. During the time period of the analysis, CBS News had the lowest compound score with both the highest negative score and the lowest negative score
3. During the time period of the analysis, CNN had the highest compound score with both the highest positive score and teh lowest negative score


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")
from datetime import datetime
from pprint import pprint
%matplotlib inline

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import api_key as consumer_key
from config import api_secret as consumer_secret
from config import access_token as access_token
from config import access_secret as access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target User Accounts
target_user = ("@BBCWorld", "@CBSNews", "@CNN", "@FoxNews", "@nytimes")

# List for results
results_list = []

# Loop through each user
for user in target_user:

    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    
    #initialize a counter @ 0
    counter = 0
    
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(1, 6):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)
        
        # Loop through all tweets
        for tweet in public_tweets:
            
            #define the objects from each tweet to later append into a list
            tweeter = tweet['user']['name']
            tweet_id = tweet['id']
            raw_time = tweet['created_at']
            converted_timestamp = datetime.strptime(raw_time, "%a %b %d %H:%M:%S %z %Y")
            text = tweet['text']
            tweet_counter = counter
            
            #Remove one from the counter
            counter -= 1
            
            #provide notification for what the program is doing
            print(f"grabbing tweet#{tweet_counter}, {tweeter}, {raw_time}")
            
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            
            #append the desired information into a list
            results_list.append([tweet_id, converted_timestamp, tweet_counter, tweeter, text, compound, pos, neu, neg])
```

    grabbing tweet#0, BBC News (World), Sat Apr 07 16:53:36 +0000 2018
    grabbing tweet#-1, BBC News (World), Sat Apr 07 15:57:19 +0000 2018
    grabbing tweet#-2, BBC News (World), Sat Apr 07 15:10:36 +0000 2018
    grabbing tweet#-3, BBC News (World), Sat Apr 07 15:07:25 +0000 2018
    grabbing tweet#-4, BBC News (World), Sat Apr 07 14:50:14 +0000 2018
    grabbing tweet#-5, BBC News (World), Sat Apr 07 14:46:08 +0000 2018
    grabbing tweet#-6, BBC News (World), Sat Apr 07 14:41:20 +0000 2018
    grabbing tweet#-7, BBC News (World), Sat Apr 07 13:58:36 +0000 2018
    grabbing tweet#-8, BBC News (World), Sat Apr 07 12:12:02 +0000 2018
    grabbing tweet#-9, BBC News (World), Sat Apr 07 12:06:31 +0000 2018
    grabbing tweet#-10, BBC News (World), Sat Apr 07 12:00:24 +0000 2018
    grabbing tweet#-11, BBC News (World), Sat Apr 07 12:00:24 +0000 2018
    grabbing tweet#-12, BBC News (World), Sat Apr 07 11:45:19 +0000 2018
    grabbing tweet#-13, BBC News (World), Sat Apr 07 11:17:16 +0000 2018
    grabbing tweet#-14, BBC News (World), Sat Apr 07 10:39:16 +0000 2018
    grabbing tweet#-15, BBC News (World), Sat Apr 07 10:30:23 +0000 2018
    grabbing tweet#-16, BBC News (World), Sat Apr 07 10:13:02 +0000 2018
    grabbing tweet#-17, BBC News (World), Sat Apr 07 08:51:03 +0000 2018
    grabbing tweet#-18, BBC News (World), Sat Apr 07 07:33:14 +0000 2018
    grabbing tweet#-19, BBC News (World), Sat Apr 07 06:11:02 +0000 2018
    grabbing tweet#-20, BBC News (World), Sat Apr 07 04:35:38 +0000 2018
    grabbing tweet#-21, BBC News (World), Sat Apr 07 03:57:43 +0000 2018
    grabbing tweet#-22, BBC News (World), Sat Apr 07 01:01:49 +0000 2018
    grabbing tweet#-23, BBC News (World), Sat Apr 07 00:28:17 +0000 2018
    grabbing tweet#-24, BBC News (World), Sat Apr 07 00:06:44 +0000 2018
    grabbing tweet#-25, BBC News (World), Fri Apr 06 23:28:33 +0000 2018
    grabbing tweet#-26, BBC News (World), Fri Apr 06 23:22:13 +0000 2018
    grabbing tweet#-27, BBC News (World), Fri Apr 06 22:14:15 +0000 2018
    grabbing tweet#-28, BBC News (World), Fri Apr 06 21:05:54 +0000 2018
    grabbing tweet#-29, BBC News (World), Fri Apr 06 19:09:19 +0000 2018
    grabbing tweet#-30, BBC News (World), Fri Apr 06 19:06:33 +0000 2018
    grabbing tweet#-31, BBC News (World), Fri Apr 06 18:10:30 +0000 2018
    grabbing tweet#-32, BBC News (World), Fri Apr 06 17:50:35 +0000 2018
    grabbing tweet#-33, BBC News (World), Fri Apr 06 16:37:30 +0000 2018
    grabbing tweet#-34, BBC News (World), Fri Apr 06 15:51:46 +0000 2018
    grabbing tweet#-35, BBC News (World), Fri Apr 06 15:50:11 +0000 2018
    grabbing tweet#-36, BBC News (World), Fri Apr 06 15:28:00 +0000 2018
    grabbing tweet#-37, BBC News (World), Fri Apr 06 15:17:38 +0000 2018
    grabbing tweet#-38, BBC News (World), Fri Apr 06 14:43:55 +0000 2018
    grabbing tweet#-39, BBC News (World), Fri Apr 06 14:17:43 +0000 2018
    grabbing tweet#-40, BBC News (World), Fri Apr 06 14:11:19 +0000 2018
    grabbing tweet#-41, BBC News (World), Fri Apr 06 14:07:25 +0000 2018
    grabbing tweet#-42, BBC News (World), Fri Apr 06 14:00:34 +0000 2018
    grabbing tweet#-43, BBC News (World), Fri Apr 06 13:45:51 +0000 2018
    grabbing tweet#-44, BBC News (World), Fri Apr 06 13:42:17 +0000 2018
    grabbing tweet#-45, BBC News (World), Fri Apr 06 13:42:17 +0000 2018
    grabbing tweet#-46, BBC News (World), Fri Apr 06 13:38:36 +0000 2018
    grabbing tweet#-47, BBC News (World), Fri Apr 06 13:38:36 +0000 2018
    grabbing tweet#-48, BBC News (World), Fri Apr 06 13:30:37 +0000 2018
    grabbing tweet#-49, BBC News (World), Fri Apr 06 13:27:00 +0000 2018
    grabbing tweet#-50, BBC News (World), Fri Apr 06 13:23:35 +0000 2018
    grabbing tweet#-51, BBC News (World), Fri Apr 06 13:06:48 +0000 2018
    grabbing tweet#-52, BBC News (World), Fri Apr 06 12:40:08 +0000 2018
    grabbing tweet#-53, BBC News (World), Fri Apr 06 12:18:56 +0000 2018
    grabbing tweet#-54, BBC News (World), Fri Apr 06 11:59:18 +0000 2018
    grabbing tweet#-55, BBC News (World), Fri Apr 06 11:24:42 +0000 2018
    grabbing tweet#-56, BBC News (World), Fri Apr 06 10:47:36 +0000 2018
    grabbing tweet#-57, BBC News (World), Fri Apr 06 10:34:42 +0000 2018
    grabbing tweet#-58, BBC News (World), Fri Apr 06 10:21:54 +0000 2018
    grabbing tweet#-59, BBC News (World), Fri Apr 06 10:19:35 +0000 2018
    grabbing tweet#-60, BBC News (World), Fri Apr 06 09:54:14 +0000 2018
    grabbing tweet#-61, BBC News (World), Fri Apr 06 09:48:49 +0000 2018
    grabbing tweet#-62, BBC News (World), Fri Apr 06 09:46:47 +0000 2018
    grabbing tweet#-63, BBC News (World), Fri Apr 06 09:07:00 +0000 2018
    grabbing tweet#-64, BBC News (World), Fri Apr 06 08:56:52 +0000 2018
    grabbing tweet#-65, BBC News (World), Fri Apr 06 08:54:44 +0000 2018
    grabbing tweet#-66, BBC News (World), Fri Apr 06 08:15:09 +0000 2018
    grabbing tweet#-67, BBC News (World), Fri Apr 06 06:54:14 +0000 2018
    grabbing tweet#-68, BBC News (World), Fri Apr 06 06:49:00 +0000 2018
    grabbing tweet#-69, BBC News (World), Fri Apr 06 06:13:01 +0000 2018
    grabbing tweet#-70, BBC News (World), Fri Apr 06 06:00:49 +0000 2018
    grabbing tweet#-71, BBC News (World), Fri Apr 06 05:48:34 +0000 2018
    grabbing tweet#-72, BBC News (World), Fri Apr 06 03:38:38 +0000 2018
    grabbing tweet#-73, BBC News (World), Fri Apr 06 03:38:38 +0000 2018
    grabbing tweet#-74, BBC News (World), Fri Apr 06 02:44:16 +0000 2018
    grabbing tweet#-75, BBC News (World), Fri Apr 06 01:34:50 +0000 2018
    grabbing tweet#-76, BBC News (World), Fri Apr 06 01:14:31 +0000 2018
    grabbing tweet#-77, BBC News (World), Fri Apr 06 00:32:20 +0000 2018
    grabbing tweet#-78, BBC News (World), Fri Apr 06 00:22:28 +0000 2018
    grabbing tweet#-79, BBC News (World), Thu Apr 05 23:59:08 +0000 2018
    grabbing tweet#-80, BBC News (World), Thu Apr 05 23:26:20 +0000 2018
    grabbing tweet#-81, BBC News (World), Thu Apr 05 23:13:50 +0000 2018
    grabbing tweet#-82, BBC News (World), Thu Apr 05 23:06:12 +0000 2018
    grabbing tweet#-83, BBC News (World), Thu Apr 05 22:08:40 +0000 2018
    grabbing tweet#-84, BBC News (World), Thu Apr 05 20:03:16 +0000 2018
    grabbing tweet#-85, BBC News (World), Thu Apr 05 19:40:45 +0000 2018
    grabbing tweet#-86, BBC News (World), Thu Apr 05 19:37:52 +0000 2018
    grabbing tweet#-87, BBC News (World), Thu Apr 05 18:36:43 +0000 2018
    grabbing tweet#-88, BBC News (World), Thu Apr 05 17:47:19 +0000 2018
    grabbing tweet#-89, BBC News (World), Thu Apr 05 17:34:23 +0000 2018
    grabbing tweet#-90, BBC News (World), Thu Apr 05 17:19:34 +0000 2018
    grabbing tweet#-91, BBC News (World), Thu Apr 05 17:12:55 +0000 2018
    grabbing tweet#-92, BBC News (World), Thu Apr 05 16:54:26 +0000 2018
    grabbing tweet#-93, BBC News (World), Thu Apr 05 16:41:02 +0000 2018
    grabbing tweet#-94, BBC News (World), Thu Apr 05 16:39:05 +0000 2018
    grabbing tweet#-95, BBC News (World), Thu Apr 05 15:51:23 +0000 2018
    grabbing tweet#-96, BBC News (World), Thu Apr 05 15:38:38 +0000 2018
    grabbing tweet#-97, BBC News (World), Thu Apr 05 15:28:55 +0000 2018
    grabbing tweet#-98, BBC News (World), Thu Apr 05 15:06:37 +0000 2018
    grabbing tweet#0, CBS News, Sat Apr 07 18:35:01 +0000 2018
    grabbing tweet#-1, CBS News, Sat Apr 07 18:17:31 +0000 2018
    grabbing tweet#-2, CBS News, Sat Apr 07 18:08:12 +0000 2018
    grabbing tweet#-3, CBS News, Sat Apr 07 17:55:02 +0000 2018
    grabbing tweet#-4, CBS News, Sat Apr 07 17:35:01 +0000 2018
    grabbing tweet#-5, CBS News, Sat Apr 07 17:16:01 +0000 2018
    grabbing tweet#-6, CBS News, Sat Apr 07 16:55:24 +0000 2018
    grabbing tweet#-7, CBS News, Sat Apr 07 16:42:01 +0000 2018
    grabbing tweet#-8, CBS News, Sat Apr 07 16:39:36 +0000 2018
    grabbing tweet#-9, CBS News, Sat Apr 07 16:25:01 +0000 2018
    grabbing tweet#-10, CBS News, Sat Apr 07 16:21:00 +0000 2018
    grabbing tweet#-11, CBS News, Sat Apr 07 16:08:07 +0000 2018
    grabbing tweet#-12, CBS News, Sat Apr 07 16:06:37 +0000 2018
    grabbing tweet#-13, CBS News, Sat Apr 07 15:43:49 +0000 2018
    grabbing tweet#-14, CBS News, Sat Apr 07 15:40:01 +0000 2018
    grabbing tweet#-15, CBS News, Sat Apr 07 15:28:36 +0000 2018
    grabbing tweet#-16, CBS News, Sat Apr 07 15:20:38 +0000 2018
    grabbing tweet#-17, CBS News, Sat Apr 07 15:12:01 +0000 2018
    grabbing tweet#-18, CBS News, Sat Apr 07 15:01:01 +0000 2018
    grabbing tweet#-19, CBS News, Sat Apr 07 14:40:01 +0000 2018
    grabbing tweet#-20, CBS News, Sat Apr 07 14:20:00 +0000 2018
    grabbing tweet#-21, CBS News, Sat Apr 07 14:00:39 +0000 2018
    grabbing tweet#-22, CBS News, Sat Apr 07 13:40:01 +0000 2018
    grabbing tweet#-23, CBS News, Sat Apr 07 13:20:01 +0000 2018
    grabbing tweet#-24, CBS News, Sat Apr 07 13:03:04 +0000 2018
    grabbing tweet#-25, CBS News, Sat Apr 07 12:48:04 +0000 2018
    grabbing tweet#-26, CBS News, Sat Apr 07 12:45:08 +0000 2018
    grabbing tweet#-27, CBS News, Sat Apr 07 12:33:04 +0000 2018
    grabbing tweet#-28, CBS News, Sat Apr 07 12:31:00 +0000 2018
    grabbing tweet#-29, CBS News, Sat Apr 07 12:30:10 +0000 2018
    grabbing tweet#-30, CBS News, Sat Apr 07 12:23:37 +0000 2018
    grabbing tweet#-31, CBS News, Sat Apr 07 12:18:03 +0000 2018
    grabbing tweet#-32, CBS News, Sat Apr 07 12:03:05 +0000 2018
    grabbing tweet#-33, CBS News, Sat Apr 07 11:54:06 +0000 2018
    grabbing tweet#-34, CBS News, Sat Apr 07 11:48:04 +0000 2018
    grabbing tweet#-35, CBS News, Sat Apr 07 11:33:03 +0000 2018
    grabbing tweet#-36, CBS News, Sat Apr 07 11:18:04 +0000 2018
    grabbing tweet#-37, CBS News, Sat Apr 07 11:03:04 +0000 2018
    grabbing tweet#-38, CBS News, Sat Apr 07 10:48:04 +0000 2018
    grabbing tweet#-39, CBS News, Sat Apr 07 10:47:00 +0000 2018
    grabbing tweet#-40, CBS News, Sat Apr 07 10:33:03 +0000 2018
    grabbing tweet#-41, CBS News, Sat Apr 07 10:18:04 +0000 2018
    grabbing tweet#-42, CBS News, Sat Apr 07 10:03:04 +0000 2018
    grabbing tweet#-43, CBS News, Sat Apr 07 09:48:03 +0000 2018
    grabbing tweet#-44, CBS News, Sat Apr 07 09:33:04 +0000 2018
    grabbing tweet#-45, CBS News, Sat Apr 07 09:18:03 +0000 2018
    grabbing tweet#-46, CBS News, Sat Apr 07 09:03:03 +0000 2018
    grabbing tweet#-47, CBS News, Sat Apr 07 08:48:04 +0000 2018
    grabbing tweet#-48, CBS News, Sat Apr 07 08:33:03 +0000 2018
    grabbing tweet#-49, CBS News, Sat Apr 07 08:18:04 +0000 2018
    grabbing tweet#-50, CBS News, Sat Apr 07 08:03:04 +0000 2018
    grabbing tweet#-51, CBS News, Sat Apr 07 07:48:03 +0000 2018
    grabbing tweet#-52, CBS News, Sat Apr 07 07:33:04 +0000 2018
    grabbing tweet#-53, CBS News, Sat Apr 07 07:18:03 +0000 2018
    grabbing tweet#-54, CBS News, Sat Apr 07 07:03:04 +0000 2018
    grabbing tweet#-55, CBS News, Sat Apr 07 06:48:03 +0000 2018
    grabbing tweet#-56, CBS News, Sat Apr 07 06:33:04 +0000 2018
    grabbing tweet#-57, CBS News, Sat Apr 07 06:18:04 +0000 2018
    grabbing tweet#-58, CBS News, Sat Apr 07 06:03:04 +0000 2018
    grabbing tweet#-59, CBS News, Sat Apr 07 05:48:04 +0000 2018
    grabbing tweet#-60, CBS News, Sat Apr 07 05:33:04 +0000 2018
    grabbing tweet#-61, CBS News, Sat Apr 07 05:18:04 +0000 2018
    grabbing tweet#-62, CBS News, Sat Apr 07 05:16:04 +0000 2018
    grabbing tweet#-63, CBS News, Sat Apr 07 05:03:05 +0000 2018
    grabbing tweet#-64, CBS News, Sat Apr 07 04:48:04 +0000 2018
    grabbing tweet#-65, CBS News, Sat Apr 07 04:46:03 +0000 2018
    grabbing tweet#-66, CBS News, Sat Apr 07 04:33:03 +0000 2018
    grabbing tweet#-67, CBS News, Sat Apr 07 04:18:04 +0000 2018
    grabbing tweet#-68, CBS News, Sat Apr 07 04:03:03 +0000 2018
    grabbing tweet#-69, CBS News, Sat Apr 07 03:48:05 +0000 2018
    grabbing tweet#-70, CBS News, Sat Apr 07 03:33:04 +0000 2018
    grabbing tweet#-71, CBS News, Sat Apr 07 03:24:04 +0000 2018
    grabbing tweet#-72, CBS News, Sat Apr 07 03:18:04 +0000 2018
    grabbing tweet#-73, CBS News, Sat Apr 07 03:03:04 +0000 2018
    grabbing tweet#-74, CBS News, Sat Apr 07 02:40:01 +0000 2018
    grabbing tweet#-75, CBS News, Sat Apr 07 02:20:01 +0000 2018
    grabbing tweet#-76, CBS News, Sat Apr 07 02:00:02 +0000 2018
    grabbing tweet#-77, CBS News, Sat Apr 07 01:40:01 +0000 2018
    grabbing tweet#-78, CBS News, Sat Apr 07 01:20:00 +0000 2018
    grabbing tweet#-79, CBS News, Sat Apr 07 00:55:01 +0000 2018
    grabbing tweet#-80, CBS News, Sat Apr 07 00:35:01 +0000 2018
    grabbing tweet#-81, CBS News, Sat Apr 07 00:15:01 +0000 2018
    grabbing tweet#-82, CBS News, Fri Apr 06 23:55:01 +0000 2018
    grabbing tweet#-83, CBS News, Fri Apr 06 23:50:44 +0000 2018
    grabbing tweet#-84, CBS News, Fri Apr 06 23:45:35 +0000 2018
    grabbing tweet#-85, CBS News, Fri Apr 06 23:35:01 +0000 2018
    grabbing tweet#-86, CBS News, Fri Apr 06 23:15:01 +0000 2018
    grabbing tweet#-87, CBS News, Fri Apr 06 22:59:00 +0000 2018
    grabbing tweet#-88, CBS News, Fri Apr 06 22:51:21 +0000 2018
    grabbing tweet#-89, CBS News, Fri Apr 06 22:49:32 +0000 2018
    grabbing tweet#-90, CBS News, Fri Apr 06 22:42:56 +0000 2018
    grabbing tweet#-91, CBS News, Fri Apr 06 22:40:57 +0000 2018
    grabbing tweet#-92, CBS News, Fri Apr 06 22:39:16 +0000 2018
    grabbing tweet#-93, CBS News, Fri Apr 06 22:36:58 +0000 2018
    grabbing tweet#-94, CBS News, Fri Apr 06 22:34:59 +0000 2018
    grabbing tweet#-95, CBS News, Fri Apr 06 22:20:01 +0000 2018
    grabbing tweet#-96, CBS News, Fri Apr 06 22:05:01 +0000 2018
    grabbing tweet#-97, CBS News, Fri Apr 06 21:50:01 +0000 2018
    grabbing tweet#-98, CBS News, Fri Apr 06 21:35:01 +0000 2018
    grabbing tweet#-99, CBS News, Fri Apr 06 21:20:33 +0000 2018
    grabbing tweet#0, CNN, Sat Apr 07 18:45:07 +0000 2018
    grabbing tweet#-1, CNN, Sat Apr 07 18:30:17 +0000 2018
    grabbing tweet#-2, CNN, Sat Apr 07 18:15:07 +0000 2018
    grabbing tweet#-3, CNN, Sat Apr 07 18:00:11 +0000 2018
    grabbing tweet#-4, CNN, Sat Apr 07 17:45:03 +0000 2018
    grabbing tweet#-5, CNN, Sat Apr 07 17:30:10 +0000 2018
    grabbing tweet#-6, CNN, Sat Apr 07 17:16:16 +0000 2018
    grabbing tweet#-7, CNN, Sat Apr 07 17:00:14 +0000 2018
    grabbing tweet#-8, CNN, Sat Apr 07 16:45:06 +0000 2018
    grabbing tweet#-9, CNN, Sat Apr 07 16:37:05 +0000 2018
    grabbing tweet#-10, CNN, Sat Apr 07 16:30:13 +0000 2018
    grabbing tweet#-11, CNN, Sat Apr 07 16:15:09 +0000 2018
    grabbing tweet#-12, CNN, Sat Apr 07 16:04:30 +0000 2018
    grabbing tweet#-13, CNN, Sat Apr 07 16:00:03 +0000 2018
    grabbing tweet#-14, CNN, Sat Apr 07 15:45:05 +0000 2018
    grabbing tweet#-15, CNN, Sat Apr 07 15:38:32 +0000 2018
    grabbing tweet#-16, CNN, Sat Apr 07 15:30:12 +0000 2018
    grabbing tweet#-17, CNN, Sat Apr 07 15:25:45 +0000 2018
    grabbing tweet#-18, CNN, Sat Apr 07 15:15:10 +0000 2018
    grabbing tweet#-19, CNN, Sat Apr 07 15:00:03 +0000 2018
    grabbing tweet#-20, CNN, Sat Apr 07 14:45:08 +0000 2018
    grabbing tweet#-21, CNN, Sat Apr 07 14:30:11 +0000 2018
    grabbing tweet#-22, CNN, Sat Apr 07 14:22:38 +0000 2018
    grabbing tweet#-23, CNN, Sat Apr 07 14:01:06 +0000 2018
    grabbing tweet#-24, CNN, Sat Apr 07 13:31:03 +0000 2018
    grabbing tweet#-25, CNN, Sat Apr 07 13:01:06 +0000 2018
    grabbing tweet#-26, CNN, Sat Apr 07 12:31:03 +0000 2018
    grabbing tweet#-27, CNN, Sat Apr 07 12:16:00 +0000 2018
    grabbing tweet#-28, CNN, Sat Apr 07 12:01:05 +0000 2018
    grabbing tweet#-29, CNN, Sat Apr 07 11:31:48 +0000 2018
    grabbing tweet#-30, CNN, Sat Apr 07 11:01:02 +0000 2018
    grabbing tweet#-31, CNN, Sat Apr 07 10:57:42 +0000 2018
    grabbing tweet#-32, CNN, Sat Apr 07 10:31:04 +0000 2018
    grabbing tweet#-33, CNN, Sat Apr 07 10:01:04 +0000 2018
    grabbing tweet#-34, CNN, Sat Apr 07 09:58:11 +0000 2018
    grabbing tweet#-35, CNN, Sat Apr 07 09:31:06 +0000 2018
    grabbing tweet#-36, CNN, Sat Apr 07 09:01:02 +0000 2018
    grabbing tweet#-37, CNN, Sat Apr 07 08:31:05 +0000 2018
    grabbing tweet#-38, CNN, Sat Apr 07 08:21:22 +0000 2018
    grabbing tweet#-39, CNN, Sat Apr 07 08:01:04 +0000 2018
    grabbing tweet#-40, CNN, Sat Apr 07 07:31:06 +0000 2018
    grabbing tweet#-41, CNN, Sat Apr 07 07:16:02 +0000 2018
    grabbing tweet#-42, CNN, Sat Apr 07 07:01:03 +0000 2018
    grabbing tweet#-43, CNN, Sat Apr 07 06:46:00 +0000 2018
    grabbing tweet#-44, CNN, Sat Apr 07 06:31:06 +0000 2018
    grabbing tweet#-45, CNN, Sat Apr 07 06:15:09 +0000 2018
    grabbing tweet#-46, CNN, Sat Apr 07 06:01:02 +0000 2018
    grabbing tweet#-47, CNN, Sat Apr 07 05:46:00 +0000 2018
    grabbing tweet#-48, CNN, Sat Apr 07 05:31:06 +0000 2018
    grabbing tweet#-49, CNN, Sat Apr 07 05:16:00 +0000 2018
    grabbing tweet#-50, CNN, Sat Apr 07 05:01:05 +0000 2018
    grabbing tweet#-51, CNN, Sat Apr 07 04:46:03 +0000 2018
    grabbing tweet#-52, CNN, Sat Apr 07 04:31:03 +0000 2018
    grabbing tweet#-53, CNN, Sat Apr 07 04:16:01 +0000 2018
    grabbing tweet#-54, CNN, Sat Apr 07 04:02:18 +0000 2018
    grabbing tweet#-55, CNN, Sat Apr 07 03:46:01 +0000 2018
    grabbing tweet#-56, CNN, Sat Apr 07 03:31:04 +0000 2018
    grabbing tweet#-57, CNN, Sat Apr 07 03:16:02 +0000 2018
    grabbing tweet#-58, CNN, Sat Apr 07 03:01:04 +0000 2018
    grabbing tweet#-59, CNN, Sat Apr 07 02:59:43 +0000 2018
    grabbing tweet#-60, CNN, Sat Apr 07 02:46:06 +0000 2018
    grabbing tweet#-61, CNN, Sat Apr 07 02:41:31 +0000 2018
    grabbing tweet#-62, CNN, Sat Apr 07 02:31:03 +0000 2018
    grabbing tweet#-63, CNN, Sat Apr 07 02:21:06 +0000 2018
    grabbing tweet#-64, CNN, Sat Apr 07 02:16:03 +0000 2018
    grabbing tweet#-65, CNN, Sat Apr 07 02:11:06 +0000 2018
    grabbing tweet#-66, CNN, Sat Apr 07 02:06:06 +0000 2018
    grabbing tweet#-67, CNN, Sat Apr 07 02:01:04 +0000 2018
    grabbing tweet#-68, CNN, Sat Apr 07 01:51:03 +0000 2018
    grabbing tweet#-69, CNN, Sat Apr 07 01:46:04 +0000 2018
    grabbing tweet#-70, CNN, Sat Apr 07 01:31:51 +0000 2018
    grabbing tweet#-71, CNN, Sat Apr 07 01:22:04 +0000 2018
    grabbing tweet#-72, CNN, Sat Apr 07 01:16:06 +0000 2018
    grabbing tweet#-73, CNN, Sat Apr 07 01:03:02 +0000 2018
    grabbing tweet#-74, CNN, Sat Apr 07 00:57:14 +0000 2018
    grabbing tweet#-75, CNN, Sat Apr 07 00:47:33 +0000 2018
    grabbing tweet#-76, CNN, Sat Apr 07 00:43:42 +0000 2018
    grabbing tweet#-77, CNN, Sat Apr 07 00:33:04 +0000 2018
    grabbing tweet#-78, CNN, Sat Apr 07 00:30:16 +0000 2018
    grabbing tweet#-79, CNN, Sat Apr 07 00:27:25 +0000 2018
    grabbing tweet#-80, CNN, Sat Apr 07 00:25:01 +0000 2018
    grabbing tweet#-81, CNN, Sat Apr 07 00:19:00 +0000 2018
    grabbing tweet#-82, CNN, Sat Apr 07 00:17:03 +0000 2018
    grabbing tweet#-83, CNN, Sat Apr 07 00:16:05 +0000 2018
    grabbing tweet#-84, CNN, Sat Apr 07 00:15:49 +0000 2018
    grabbing tweet#-85, CNN, Sat Apr 07 00:07:04 +0000 2018
    grabbing tweet#-86, CNN, Sat Apr 07 00:04:08 +0000 2018
    grabbing tweet#-87, CNN, Sat Apr 07 00:03:33 +0000 2018
    grabbing tweet#-88, CNN, Fri Apr 06 23:58:21 +0000 2018
    grabbing tweet#-89, CNN, Fri Apr 06 23:53:02 +0000 2018
    grabbing tweet#-90, CNN, Fri Apr 06 23:47:59 +0000 2018
    grabbing tweet#-91, CNN, Fri Apr 06 23:41:21 +0000 2018
    grabbing tweet#-92, CNN, Fri Apr 06 23:25:17 +0000 2018
    grabbing tweet#-93, CNN, Fri Apr 06 23:23:45 +0000 2018
    grabbing tweet#-94, CNN, Fri Apr 06 23:23:36 +0000 2018
    grabbing tweet#-95, CNN, Fri Apr 06 23:15:08 +0000 2018
    grabbing tweet#-96, CNN, Fri Apr 06 23:07:50 +0000 2018
    grabbing tweet#-97, CNN, Fri Apr 06 22:59:43 +0000 2018
    grabbing tweet#-98, CNN, Fri Apr 06 22:58:00 +0000 2018
    grabbing tweet#-99, CNN, Fri Apr 06 22:55:15 +0000 2018
    grabbing tweet#0, Fox News, Sat Apr 07 18:48:49 +0000 2018
    grabbing tweet#-1, Fox News, Sat Apr 07 18:39:54 +0000 2018
    grabbing tweet#-2, Fox News, Sat Apr 07 18:37:58 +0000 2018
    grabbing tweet#-3, Fox News, Sat Apr 07 18:34:05 +0000 2018
    grabbing tweet#-4, Fox News, Sat Apr 07 18:29:23 +0000 2018
    grabbing tweet#-5, Fox News, Sat Apr 07 18:28:00 +0000 2018
    grabbing tweet#-6, Fox News, Sat Apr 07 18:27:00 +0000 2018
    grabbing tweet#-7, Fox News, Sat Apr 07 18:26:19 +0000 2018
    grabbing tweet#-8, Fox News, Sat Apr 07 18:16:46 +0000 2018
    grabbing tweet#-9, Fox News, Sat Apr 07 18:10:51 +0000 2018
    grabbing tweet#-10, Fox News, Sat Apr 07 17:56:51 +0000 2018
    grabbing tweet#-11, Fox News, Sat Apr 07 17:46:35 +0000 2018
    grabbing tweet#-12, Fox News, Sat Apr 07 17:44:27 +0000 2018
    grabbing tweet#-13, Fox News, Sat Apr 07 17:33:14 +0000 2018
    grabbing tweet#-14, Fox News, Sat Apr 07 17:27:00 +0000 2018
    grabbing tweet#-15, Fox News, Sat Apr 07 17:26:00 +0000 2018
    grabbing tweet#-16, Fox News, Sat Apr 07 17:24:00 +0000 2018
    grabbing tweet#-17, Fox News, Sat Apr 07 17:23:56 +0000 2018
    grabbing tweet#-18, Fox News, Sat Apr 07 17:16:00 +0000 2018
    grabbing tweet#-19, Fox News, Sat Apr 07 17:10:00 +0000 2018
    grabbing tweet#-20, Fox News, Sat Apr 07 17:00:03 +0000 2018
    grabbing tweet#-21, Fox News, Sat Apr 07 16:53:00 +0000 2018
    grabbing tweet#-22, Fox News, Sat Apr 07 16:52:03 +0000 2018
    grabbing tweet#-23, Fox News, Sat Apr 07 16:43:00 +0000 2018
    grabbing tweet#-24, Fox News, Sat Apr 07 16:42:55 +0000 2018
    grabbing tweet#-25, Fox News, Sat Apr 07 16:35:00 +0000 2018
    grabbing tweet#-26, Fox News, Sat Apr 07 16:27:56 +0000 2018
    grabbing tweet#-27, Fox News, Sat Apr 07 16:25:00 +0000 2018
    grabbing tweet#-28, Fox News, Sat Apr 07 16:18:58 +0000 2018
    grabbing tweet#-29, Fox News, Sat Apr 07 16:13:35 +0000 2018
    grabbing tweet#-30, Fox News, Sat Apr 07 16:06:34 +0000 2018
    grabbing tweet#-31, Fox News, Sat Apr 07 15:58:08 +0000 2018
    grabbing tweet#-32, Fox News, Sat Apr 07 15:50:36 +0000 2018
    grabbing tweet#-33, Fox News, Sat Apr 07 15:44:41 +0000 2018
    grabbing tweet#-34, Fox News, Sat Apr 07 15:34:15 +0000 2018
    grabbing tweet#-35, Fox News, Sat Apr 07 15:21:48 +0000 2018
    grabbing tweet#-36, Fox News, Sat Apr 07 15:20:46 +0000 2018
    grabbing tweet#-37, Fox News, Sat Apr 07 15:19:43 +0000 2018
    grabbing tweet#-38, Fox News, Sat Apr 07 15:16:09 +0000 2018
    grabbing tweet#-39, Fox News, Sat Apr 07 15:02:44 +0000 2018
    grabbing tweet#-40, Fox News, Sat Apr 07 14:55:00 +0000 2018
    grabbing tweet#-41, Fox News, Sat Apr 07 14:49:00 +0000 2018
    grabbing tweet#-42, Fox News, Sat Apr 07 14:40:59 +0000 2018
    grabbing tweet#-43, Fox News, Sat Apr 07 14:33:40 +0000 2018
    grabbing tweet#-44, Fox News, Sat Apr 07 14:28:00 +0000 2018
    grabbing tweet#-45, Fox News, Sat Apr 07 14:17:13 +0000 2018
    grabbing tweet#-46, Fox News, Sat Apr 07 14:15:53 +0000 2018
    grabbing tweet#-47, Fox News, Sat Apr 07 14:14:31 +0000 2018
    grabbing tweet#-48, Fox News, Sat Apr 07 14:13:30 +0000 2018
    grabbing tweet#-49, Fox News, Sat Apr 07 14:12:57 +0000 2018
    grabbing tweet#-50, Fox News, Sat Apr 07 14:11:57 +0000 2018
    grabbing tweet#-51, Fox News, Sat Apr 07 14:11:18 +0000 2018
    grabbing tweet#-52, Fox News, Sat Apr 07 14:10:40 +0000 2018
    grabbing tweet#-53, Fox News, Sat Apr 07 14:06:26 +0000 2018
    grabbing tweet#-54, Fox News, Sat Apr 07 13:55:33 +0000 2018
    grabbing tweet#-55, Fox News, Sat Apr 07 13:54:52 +0000 2018
    grabbing tweet#-56, Fox News, Sat Apr 07 12:42:26 +0000 2018
    grabbing tweet#-57, Fox News, Sat Apr 07 12:30:50 +0000 2018
    grabbing tweet#-58, Fox News, Sat Apr 07 12:25:19 +0000 2018
    grabbing tweet#-59, Fox News, Sat Apr 07 11:41:09 +0000 2018
    grabbing tweet#-60, Fox News, Sat Apr 07 11:12:20 +0000 2018
    grabbing tweet#-61, Fox News, Sat Apr 07 11:04:06 +0000 2018
    grabbing tweet#-62, Fox News, Sat Apr 07 10:25:05 +0000 2018
    grabbing tweet#-63, Fox News, Sat Apr 07 10:22:18 +0000 2018
    grabbing tweet#-64, Fox News, Sat Apr 07 10:16:42 +0000 2018
    grabbing tweet#-65, Fox News, Sat Apr 07 10:15:00 +0000 2018
    grabbing tweet#-66, Fox News, Sat Apr 07 10:06:01 +0000 2018
    grabbing tweet#-67, Fox News, Sat Apr 07 10:00:02 +0000 2018
    grabbing tweet#-68, Fox News, Sat Apr 07 09:45:00 +0000 2018
    grabbing tweet#-69, Fox News, Sat Apr 07 09:30:00 +0000 2018
    grabbing tweet#-70, Fox News, Sat Apr 07 09:15:00 +0000 2018
    grabbing tweet#-71, Fox News, Sat Apr 07 09:00:01 +0000 2018
    grabbing tweet#-72, Fox News, Sat Apr 07 08:45:00 +0000 2018
    grabbing tweet#-73, Fox News, Sat Apr 07 08:30:00 +0000 2018
    grabbing tweet#-74, Fox News, Sat Apr 07 08:19:00 +0000 2018
    grabbing tweet#-75, Fox News, Sat Apr 07 08:16:00 +0000 2018
    grabbing tweet#-76, Fox News, Sat Apr 07 08:05:00 +0000 2018
    grabbing tweet#-77, Fox News, Sat Apr 07 07:50:19 +0000 2018
    grabbing tweet#-78, Fox News, Sat Apr 07 07:48:00 +0000 2018
    grabbing tweet#-79, Fox News, Sat Apr 07 07:33:00 +0000 2018
    grabbing tweet#-80, Fox News, Sat Apr 07 07:15:00 +0000 2018
    grabbing tweet#-81, Fox News, Sat Apr 07 07:00:00 +0000 2018
    grabbing tweet#-82, Fox News, Sat Apr 07 06:45:00 +0000 2018
    grabbing tweet#-83, Fox News, Sat Apr 07 06:27:00 +0000 2018
    grabbing tweet#-84, Fox News, Sat Apr 07 06:18:00 +0000 2018
    grabbing tweet#-85, Fox News, Sat Apr 07 06:00:00 +0000 2018
    grabbing tweet#-86, Fox News, Sat Apr 07 05:48:00 +0000 2018
    grabbing tweet#-87, Fox News, Sat Apr 07 05:37:00 +0000 2018
    grabbing tweet#-88, Fox News, Sat Apr 07 05:00:00 +0000 2018
    grabbing tweet#-89, Fox News, Sat Apr 07 04:50:00 +0000 2018
    grabbing tweet#-90, Fox News, Sat Apr 07 04:46:00 +0000 2018
    grabbing tweet#-91, Fox News, Sat Apr 07 04:35:00 +0000 2018
    grabbing tweet#-92, Fox News, Sat Apr 07 04:15:00 +0000 2018
    grabbing tweet#-93, Fox News, Sat Apr 07 04:00:01 +0000 2018
    grabbing tweet#-94, Fox News, Sat Apr 07 03:57:44 +0000 2018
    grabbing tweet#-95, Fox News, Sat Apr 07 03:55:18 +0000 2018
    grabbing tweet#-96, Fox News, Sat Apr 07 03:41:12 +0000 2018
    grabbing tweet#-97, Fox News, Sat Apr 07 03:30:22 +0000 2018
    grabbing tweet#-98, Fox News, Sat Apr 07 03:12:13 +0000 2018
    grabbing tweet#-99, Fox News, Sat Apr 07 03:09:55 +0000 2018
    grabbing tweet#0, The New York Times, Sat Apr 07 18:47:06 +0000 2018
    grabbing tweet#-1, The New York Times, Sat Apr 07 18:34:03 +0000 2018
    grabbing tweet#-2, The New York Times, Sat Apr 07 18:32:06 +0000 2018
    grabbing tweet#-3, The New York Times, Sat Apr 07 18:27:21 +0000 2018
    grabbing tweet#-4, The New York Times, Sat Apr 07 18:02:05 +0000 2018
    grabbing tweet#-5, The New York Times, Sat Apr 07 17:54:53 +0000 2018
    grabbing tweet#-6, The New York Times, Sat Apr 07 17:36:47 +0000 2018
    grabbing tweet#-7, The New York Times, Sat Apr 07 17:18:37 +0000 2018
    grabbing tweet#-8, The New York Times, Sat Apr 07 17:02:04 +0000 2018
    grabbing tweet#-9, The New York Times, Sat Apr 07 16:47:02 +0000 2018
    grabbing tweet#-10, The New York Times, Sat Apr 07 16:32:06 +0000 2018
    grabbing tweet#-11, The New York Times, Sat Apr 07 16:17:04 +0000 2018
    grabbing tweet#-12, The New York Times, Sat Apr 07 16:02:05 +0000 2018
    grabbing tweet#-13, The New York Times, Sat Apr 07 15:47:03 +0000 2018
    grabbing tweet#-14, The New York Times, Sat Apr 07 15:32:02 +0000 2018
    grabbing tweet#-15, The New York Times, Sat Apr 07 15:17:04 +0000 2018
    grabbing tweet#-16, The New York Times, Sat Apr 07 15:02:03 +0000 2018
    grabbing tweet#-17, The New York Times, Sat Apr 07 14:47:05 +0000 2018
    grabbing tweet#-18, The New York Times, Sat Apr 07 14:32:04 +0000 2018
    grabbing tweet#-19, The New York Times, Sat Apr 07 14:17:04 +0000 2018
    grabbing tweet#-20, The New York Times, Sat Apr 07 14:02:09 +0000 2018
    grabbing tweet#-21, The New York Times, Sat Apr 07 13:47:44 +0000 2018
    grabbing tweet#-22, The New York Times, Sat Apr 07 13:41:06 +0000 2018
    grabbing tweet#-23, The New York Times, Sat Apr 07 13:31:00 +0000 2018
    grabbing tweet#-24, The New York Times, Sat Apr 07 13:21:03 +0000 2018
    grabbing tweet#-25, The New York Times, Sat Apr 07 13:02:07 +0000 2018
    grabbing tweet#-26, The New York Times, Sat Apr 07 12:41:06 +0000 2018
    grabbing tweet#-27, The New York Times, Sat Apr 07 12:31:08 +0000 2018
    grabbing tweet#-28, The New York Times, Sat Apr 07 12:21:02 +0000 2018
    grabbing tweet#-29, The New York Times, Sat Apr 07 12:02:09 +0000 2018
    grabbing tweet#-30, The New York Times, Sat Apr 07 11:55:33 +0000 2018
    grabbing tweet#-31, The New York Times, Sat Apr 07 11:44:00 +0000 2018
    grabbing tweet#-32, The New York Times, Sat Apr 07 11:41:04 +0000 2018
    grabbing tweet#-33, The New York Times, Sat Apr 07 11:21:04 +0000 2018
    grabbing tweet#-34, The New York Times, Sat Apr 07 11:02:03 +0000 2018
    grabbing tweet#-35, The New York Times, Sat Apr 07 10:41:09 +0000 2018
    grabbing tweet#-36, The New York Times, Sat Apr 07 10:25:01 +0000 2018
    grabbing tweet#-37, The New York Times, Sat Apr 07 10:09:53 +0000 2018
    grabbing tweet#-38, The New York Times, Sat Apr 07 09:51:24 +0000 2018
    grabbing tweet#-39, The New York Times, Sat Apr 07 09:43:02 +0000 2018
    grabbing tweet#-40, The New York Times, Sat Apr 07 09:39:49 +0000 2018
    grabbing tweet#-41, The New York Times, Sat Apr 07 09:22:39 +0000 2018
    grabbing tweet#-42, The New York Times, Sat Apr 07 09:06:27 +0000 2018
    grabbing tweet#-43, The New York Times, Sat Apr 07 08:56:48 +0000 2018
    grabbing tweet#-44, The New York Times, Sat Apr 07 08:38:04 +0000 2018
    grabbing tweet#-45, The New York Times, Sat Apr 07 08:17:36 +0000 2018
    grabbing tweet#-46, The New York Times, Sat Apr 07 07:58:42 +0000 2018
    grabbing tweet#-47, The New York Times, Sat Apr 07 07:40:35 +0000 2018
    grabbing tweet#-48, The New York Times, Sat Apr 07 07:22:03 +0000 2018
    grabbing tweet#-49, The New York Times, Sat Apr 07 07:07:24 +0000 2018
    grabbing tweet#-50, The New York Times, Sat Apr 07 06:48:27 +0000 2018
    grabbing tweet#-51, The New York Times, Sat Apr 07 06:34:11 +0000 2018
    grabbing tweet#-52, The New York Times, Sat Apr 07 06:17:02 +0000 2018
    grabbing tweet#-53, The New York Times, Sat Apr 07 05:56:52 +0000 2018
    grabbing tweet#-54, The New York Times, Sat Apr 07 05:38:47 +0000 2018
    grabbing tweet#-55, The New York Times, Sat Apr 07 05:22:04 +0000 2018
    grabbing tweet#-56, The New York Times, Sat Apr 07 04:55:29 +0000 2018
    grabbing tweet#-57, The New York Times, Sat Apr 07 04:45:15 +0000 2018
    grabbing tweet#-58, The New York Times, Sat Apr 07 04:27:06 +0000 2018
    grabbing tweet#-59, The New York Times, Sat Apr 07 04:09:22 +0000 2018
    grabbing tweet#-60, The New York Times, Sat Apr 07 04:03:38 +0000 2018
    grabbing tweet#-61, The New York Times, Sat Apr 07 04:02:02 +0000 2018
    grabbing tweet#-62, The New York Times, Sat Apr 07 03:47:02 +0000 2018
    grabbing tweet#-63, The New York Times, Sat Apr 07 03:33:01 +0000 2018
    grabbing tweet#-64, The New York Times, Sat Apr 07 03:25:24 +0000 2018
    grabbing tweet#-65, The New York Times, Sat Apr 07 03:17:02 +0000 2018
    grabbing tweet#-66, The New York Times, Sat Apr 07 03:02:04 +0000 2018
    grabbing tweet#-67, The New York Times, Sat Apr 07 02:47:08 +0000 2018
    grabbing tweet#-68, The New York Times, Sat Apr 07 02:32:05 +0000 2018
    grabbing tweet#-69, The New York Times, Sat Apr 07 02:15:08 +0000 2018
    grabbing tweet#-70, The New York Times, Sat Apr 07 02:02:06 +0000 2018
    grabbing tweet#-71, The New York Times, Sat Apr 07 01:47:03 +0000 2018
    grabbing tweet#-72, The New York Times, Sat Apr 07 01:33:04 +0000 2018
    grabbing tweet#-73, The New York Times, Sat Apr 07 01:17:02 +0000 2018
    grabbing tweet#-74, The New York Times, Sat Apr 07 01:02:07 +0000 2018
    grabbing tweet#-75, The New York Times, Sat Apr 07 00:47:03 +0000 2018
    grabbing tweet#-76, The New York Times, Sat Apr 07 00:32:05 +0000 2018
    grabbing tweet#-77, The New York Times, Sat Apr 07 00:17:02 +0000 2018
    grabbing tweet#-78, The New York Times, Sat Apr 07 00:02:02 +0000 2018
    grabbing tweet#-79, The New York Times, Fri Apr 06 23:47:06 +0000 2018
    grabbing tweet#-80, The New York Times, Fri Apr 06 23:32:05 +0000 2018
    grabbing tweet#-81, The New York Times, Fri Apr 06 23:16:04 +0000 2018
    grabbing tweet#-82, The New York Times, Fri Apr 06 23:06:02 +0000 2018
    grabbing tweet#-83, The New York Times, Fri Apr 06 22:56:06 +0000 2018
    grabbing tweet#-84, The New York Times, Fri Apr 06 22:43:05 +0000 2018
    grabbing tweet#-85, The New York Times, Fri Apr 06 22:32:06 +0000 2018
    grabbing tweet#-86, The New York Times, Fri Apr 06 22:17:05 +0000 2018
    grabbing tweet#-87, The New York Times, Fri Apr 06 22:02:03 +0000 2018
    grabbing tweet#-88, The New York Times, Fri Apr 06 21:47:08 +0000 2018
    grabbing tweet#-89, The New York Times, Fri Apr 06 21:32:03 +0000 2018
    grabbing tweet#-90, The New York Times, Fri Apr 06 21:15:05 +0000 2018
    grabbing tweet#-91, The New York Times, Fri Apr 06 21:05:02 +0000 2018
    grabbing tweet#-92, The New York Times, Fri Apr 06 20:55:02 +0000 2018
    grabbing tweet#-93, The New York Times, Fri Apr 06 20:45:03 +0000 2018
    grabbing tweet#-94, The New York Times, Fri Apr 06 20:32:06 +0000 2018
    grabbing tweet#-95, The New York Times, Fri Apr 06 20:19:04 +0000 2018
    grabbing tweet#-96, The New York Times, Fri Apr 06 20:15:05 +0000 2018
    grabbing tweet#-97, The New York Times, Fri Apr 06 20:00:13 +0000 2018
    grabbing tweet#-98, The New York Times, Fri Apr 06 19:51:30 +0000 2018
    grabbing tweet#-99, The New York Times, Fri Apr 06 19:45:05 +0000 2018
    


```python
#create a dataframe from the results
results_df = pd.DataFrame(results_list, columns = ["Tweet_ID", "Converted_Timestamp", "Tweet_Counter", "Tweeter", "Text", "Compound_Score", "Positive_Score", "Neutral_Score", "Negative_Score"])
results_df.to_csv("twitter_results.csv")
```


```python
#Create first plot
results_scatter = sns.pointplot(data = results_df, x = "Tweet_Counter", y = "Compound_Score", hue = "Tweeter", linestyles = "")
results_scatter = plt.title("Compound Sentiment Score vs. Time (Date: April 7, 2018)")
results_scatter = plt.xlabel("Tweets Ago")
results_scatter = plt.xticks(range(0,101,10), np.arange(-100,1,10))
results_scatter = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#save plot to a png file
plt.savefig("Twitter Figure 1 - Scatter Plot.png")
```


![png](output_5_0.png)



```python
#Creating dataframe to be used by the bar graph
scores_groupby_Tweeter = results_df[['Compound_Score', 'Positive_Score', 'Neutral_Score', 'Negative_Score']].groupby(results_df['Tweeter'])
avg_scores_Tweeter = scores_groupby_Tweeter.mean()
avg_scores_Tweeter.reset_index(inplace = True)
avg_scores_Tweeter
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweeter</th>
      <th>Compound_Score</th>
      <th>Positive_Score</th>
      <th>Neutral_Score</th>
      <th>Negative_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC News (World)</td>
      <td>-0.123284</td>
      <td>0.054545</td>
      <td>0.828313</td>
      <td>0.117141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS News</td>
      <td>-0.242478</td>
      <td>0.037010</td>
      <td>0.820630</td>
      <td>0.142340</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>-0.057105</td>
      <td>0.061470</td>
      <td>0.853880</td>
      <td>0.084630</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox News</td>
      <td>-0.142145</td>
      <td>0.069400</td>
      <td>0.795830</td>
      <td>0.134810</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The New York Times</td>
      <td>-0.007901</td>
      <td>0.075900</td>
      <td>0.840690</td>
      <td>0.083400</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creating the bar chart
results_bar = sns.barplot(x = "Tweeter", y = "Compound_Score", data = avg_scores_Tweeter)
results_bar = plt.xticks(rotation = 45)
results_bar = plt.title("Average Compound Sentiment Score by Tweeter (Date: April 7, 2018)")

#save plot to a png file
plt.savefig("Twitter Figure 2 - Boxplot.png")
```


![png](output_7_0.png)

