from datetime import datetime
import tweepy
import pandas as pd
import csv 
import re
from filter_prep import prepare_data

timestamp = 1483271920

dt = datetime.fromtimestamp(timestamp)

print (dt)


api_key = "LZ5N8bF2RWGOLuaQ6n8shjLon"
api_key_secret = "urhvy4gt9n9PbA3cORO2GXyFh2crka4qAX9vhkxScPZU1xMPel"
access_token= "1242795460009562118-hAgKdoc08XEnKofJGEEOsmKz9Jw3EK"
access_token_secret = "gdMkOOFbqzvATVnyEQDlkUz308qj1TYoYrD3NQLiPOnRW"

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

dataset = pd.read_csv('dataset/tweet_ids.txt', sep='\t', header=0)
tweet_ids = dataset.to_numpy()
corpus = [] # Collection of tweet text 


#  FETECHING TWEETS
# for id in tweet_ids[0:10]:
#     twt_id = id[0]

#     print (twt_id)
#     try:
#         print ("reading")
#         print (int(twt_id))
        
#         tweet_fetched = api.get_status(int(twt_id))
#         corpus.append(tweet_fetched.text)
#     except tweepy.errors.HTTPException as e:
#         print(e)


# PRCESSING TWEETS (FILTERING WORDS)
sentence = "in the Beginning Congestion was the Word and the word   god "
filter = ["beginning", "word"]



#-------------------- FILTERING TWEETS
tweets_df = pd.read_csv("fetched_tweets.csv", sep=',', header=0) # reading feteched tweets
test_df = pd.read_csv("Book2.csv", sep=',', header=0) 
print (test_df.info())
filter_words, event_time, reported_time = prepare_data("dataset/m25_sensors_data_all.csv") # filter data
print (len(filter_words.keys()))
matching_tweets = {}
for index, row in test_df.iterrows():

  tweet = row['Tweeet']
  # dt_object = datetime.fromtimestamp(row["Created time"])
  # print ("timeee:", dt_object)
  for key in filter_words:
    matched = [] 
    # Checking filter words

    for word in filter_words[key]:

      matched.append(word.lower() in tweet.lower())
    # print (matched)

    # Checking time
    if all(matched):
      tweet_time = row["Created time"]
      if tweet_time > event_time[key][0] and tweet_time < event_time[key][1]:
        matching_tweets[key] = [tweet, tweet_time]
        continue

print ("Number of mathcessss: ")

print (len(matching_tweets.keys()))    
print ("sucesssssssss")
  
  



# print (tweets_fetched[0])


'''Search multiple keywords '''

# print (re.findall(pattern, text, flags=re.IGNORECASE))
# if all(re.findall(pattern, text, flags=re.IGNORECASE)):
#   print ("FOUNDD")
# else:
#   print ("NOT FOUND")

# print("Tweet fetched" + tweetFetched.text)

# if any(re.findall(myList, str, re.IGNORECASE)):
#   print("Found a match")
# else:
#   print("Not Found a match")
