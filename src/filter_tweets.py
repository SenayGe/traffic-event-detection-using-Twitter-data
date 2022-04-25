import pandas as pd 
from filter_prep import prepare_data
from datetime import datetime
import string

def printer(matching_tweets,filter_words,event_time,reported_time):
  print ("------------------------ Matching tweets --------------------------------")
  iter=1
  for key in matching_tweets.keys():
    print ("--------------------------------",iter," Matching Event ",key," --------------------------------")
    print (" Event ID : ",key, "  ",matching_tweets[key][0])
    print ( datetime.fromtimestamp(matching_tweets[key][1]))
    print ("Filter Word : ",filter_words[key])
    print ("Event Start Time: ",datetime.fromtimestamp(event_time[key][0]))
    print ("Incident reported_time Start: ",datetime.fromtimestamp(reported_time[key][0]))
    print ("Incident reported_time End: ",datetime.fromtimestamp(reported_time[key][1]))
    print ("Event End Time ",datetime.fromtimestamp(event_time[key][1]))
    iter+=1
  print ("\nNumber of matchs: ",len(matching_tweets.keys())) 
  print ("\nNumber of Event: ",len(reported_time.keys())) 

#-------------------- FILTERING TWEETS ------------------------
tweets_df = pd.read_csv("Data/fetched_tweets.csv", sep=',', header=0) # reading feteched tweets
# test_df = pd.read_csv("Data/books.csv", sep=',', header=0) 
# print (test_df.info())
filter_words, event_time, reported_time = prepare_data("Data/m25_sensors_data.csv") # filter data
matching_tweets = {}
for index, row in tweets_df.iterrows():
  # print ("Index: ", index)
  tweet = row['Tweeet'].translate(str.maketrans('', '', string.punctuation)) # remove punctuation
  # dt_object = datetime.fromtimestamp(row["Created time"])
  # print ("timeee:", dt_object)
  event = None
  for key in filter_words:
    event = key
    matched = [] 
    # Checking filter words
    for word in filter_words[key]:
      matched.append(word.lower() in tweet.lower().split())

    # Checking time
    if all(matched):
      tweet_time = row["Created time"]
      # if tweet_time > event_time[key][0] and tweet_time < event_time[key][1]:
      #   matching_tweets[key] = [tweet, tweet_time]
      #   continue
      if key==298:
        continue
      if tweet_time > event_time[key][0] and tweet_time < reported_time[key][1]:
        if key in matching_tweets:
          
          pass
          # print("----------------------Repeated",key,"--------------------------")

          # print("previous data : ",matching_tweets[key][0]," ",(datetime.fromtimestamp(matching_tweets[key][1])))

          # print("New data : ",tweet," ",datetime.fromtimestamp(tweet_time))

          # print("Event data : "," ",datetime.fromtimestamp(reported_time[key][0]))

          # matching_tweets[key].append([tweet, tweet_time])

        else:

          matching_tweets[key] = [tweet, tweet_time]
        continue
      

  # analyzing unmatched tweets 
  
  if not event in matching_tweets:
    print ("----------- UNMATCHED TWEETS ------------")
    print ("Tweet: ", tweet)
    print (matched)
    print (filter_words[event])
printer(matching_tweets,filter_words,event_time,reported_time)
print ("\nSUCCESS!")
print ("\nTweets Info: "),
print (tweets_df.info(),sep='\n')
print (matching_tweets.keys())
print (filter_words[58])

