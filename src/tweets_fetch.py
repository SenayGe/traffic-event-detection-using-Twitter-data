from datetime import datetime
import tweepy
import pandas as pd
import csv
from pytz import timezone
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key'] 
api_key_secret = config['twitter']['api_key_secret'] 
access_token= config ['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


# Saving tweets to csv file
header = ["Tweeet", "Created time"]
f = open("/Users/sg/Documents/UNI/2-Software-Eng/Project/fetched_tweets-2.csv", 'w')

# create the csv writer
writer = csv.writer(f)
writer.writerow(header)

dataset = pd.read_csv('dataset/tweet_ids.txt', sep='\t', header=0)
tweet_ids = dataset.to_numpy()
corpus = [] # Collection of tweet text

for id in tweet_ids[4323:5000]:
    twt_id = id[0]

    print (twt_id)
    try:
        tweetFetched = api.get_status(int(twt_id))
        corpus.append(tweetFetched.text)
        
        tweet_time = tweetFetched.created_at.astimezone(timezone('Asia/Dubai'))
        # tweet_time = tweet_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.timestamp(tweet_time)
        writer.writerow([tweetFetched.text, timestamp])

    except tweepy.errors.HTTPException as e:
        print(e)

f.close()