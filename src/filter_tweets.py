import pandas as pd 
from filter_prep import prepare_data
from datetime import datetime
import string
import statistics as st


def add_tweet(df,matching_tweets,lifetime=287): # default life time of a tweet
# add a new feature with timestamp from tweeter
# set the max lifetime of a tweet 
  max_lifetime = lifetime # minutes
  tweet_feature =[]
  binary_feature =[]  # saves binary tweet data
  for index, row in df.iterrows():
    event_id = row['EventID']
    # find the matching tweet event
    # check when the tweet arrived
    tweet_time = datetime.fromtimestamp(matching_tweets[event_id][1])
    row_time = datetime.fromtimestamp(row['TimeStamp'])
    diff_min = int((row_time - tweet_time).total_seconds())/60
    #print(diff_min)
    # diff_min = int (abs((tweet_time-row_time).total_seconds())/60)
    # if tweet came after the row timestamp and is not older than max_lifetime then it is relevant
    if  row_time> tweet_time and abs(diff_min) < max_lifetime:
      # if index< 500 :
      #   print(index," tweet_time > row_time: " ,tweet_time ,row_time)
      tweet_feature.append(max_lifetime-diff_min)
      binary_feature.append(int(1))
    else:
      tweet_feature.append(0)
      binary_feature.append(int(0))

  return tweet_feature,binary_feature

def event_properties(matching_tweets,reported_time,printing=False):
    # Find the average time of an incident (since the reported time until it is considered not happening)
    diff_minutes = []



    for key in matching_tweets:
      start = datetime.fromtimestamp(reported_time[key][0])
      end = datetime.fromtimestamp(reported_time[key][1])
      diff_min = int((end-start).total_seconds()/60) # time diff in minutes
      diff_minutes.append(diff_min)
    average = int (sum(diff_minutes)/len(diff_minutes))

    if (printing):
      print('The average time for an event is :'+ str (average) + "   No. of events: " + str(len(diff_minutes))) 
      print('The Median time for an event is :',st.median(diff_minutes))
      print('The Max time for an event is :',max(diff_minutes))
      print('The Min time for an event is :',min(diff_minutes))
    return diff_minutes
def remove_unmached(df,unmatched):
  # Remove unmached events 
  for key in unmatched:
    # remove rows with events that are unmached
    df = df[df['EventID']!=key]
  print("Number of Matched events: ",len(df['EventID'].unique()))
  return df


def unmatched_events(filter_words,matching_tweets):
  i =0
  unmatched = []
  for key in filter_words:
    if key in matching_tweets.keys():
      pass
    else:
      unmatched.append(key)
      i+=1
     # print(i," UnMatched Event: ",key)
  return unmatched
def printer(matching_tweets,filter_words,event_time,reported_time):
  print ("------------------------ Matching tweets --------------------------------")
  iter=1
  for key in matching_tweets.keys():
    # print ("--------------------------------",iter," Matching Event ",key," --------------------------------")
    # print (" Event ID : ",key, "  ",matching_tweets[key][0])
    # print ( datetime.fromtimestamp(matching_tweets[key][1]))
    # print ("Filter Word : ",filter_words[key])
    # print ("Event Start Time: ",datetime.fromtimestamp(event_time[key][0]))
    # print ("Incident reported_time Start: ",datetime.fromtimestamp(reported_time[key][0]))
    # print ("Incident reported_time End: ",datetime.fromtimestamp(reported_time[key][1]))
    # print ("Event End Time ",datetime.fromtimestamp(event_time[key][1]))
    iter+=1
  # print ("\nNumber of matchs: ",len(matching_tweets.keys())) 
  # print ("\nNumber of Events: ",len(event_time.keys())) 

def database_creation(lifetime=278):  #-------------------- FILTERING TWEETS ------------------------
    tweets_df = pd.read_csv("Data/fetched_tweets.csv", sep=',', header=0) # reading feteched tweets
    # test_df = pd.read_csv("Data/books.csv", sep=',', header=0) 
    # print (test_df.info())
    filter_words, event_time, reported_time,df = prepare_data("Data/m25_sensors_data.csv") # filter data
    matching_tweets = {}
    for index, row in tweets_df.iterrows():
      #print ("Index: ", index)
      tweet = row['Tweeet'].translate(str.maketrans('', '', string.punctuation))


      # dt_object = datetime.fromtimestamp(row["Created time"])
      # print ("timeee:", dt_object)
      for key in filter_words:
        matched = [] 
        # Checking filter words

        for word in filter_words[key]:

          matched.append(word.lower() in tweet.lower().split())
        # print (matched)

        # Checking time
        if all(matched):
          tweet_time = row["Created time"]
          # if tweet_time > event_time[key][0] and tweet_time < event_time[key][1]:
          #   matching_tweets[key] = [tweet, tweet_time]
          #   continue
          if key not in reported_time.keys() :  # tweet found but no incident reported
            #print(key," Possible Event without incident found.......")
            if tweet_time > event_time[key][0] and tweet_time < event_time[key][1]:
              if (key in matching_tweets):
                pass
              else:
                matching_tweets[key] = [tweet, tweet_time]
                print("Event without incident found.......")
          elif tweet_time > event_time[key][0] and tweet_time < reported_time[key][1]:
            if (key in matching_tweets):  # if key already in matching ignore
              #print("----------------------Repeated",key,"--------------------------")
              # print("previous data : ",matching_tweets[key][0]," ",(datetime.fromtimestamp(matching_tweets[key][1])))
              # print("New data : ",tweet," ",datetime.fromtimestamp(tweet_time))
              # print("Event data : ",filter_words[key]," ",datetime.fromtimestamp(reported_time[key][0]))
              # matching_tweets[key].append([tweet, tweet_time])
              #print("New AFTER appending : ",matching_tweets[key])
              pass
            else:
              matching_tweets[key] = [tweet, tweet_time]
            
    print ("\nSUCCESS!")
    #print ("\nTweets Info: ")
    #print (tweets_df.info(),sep='\n')
    #printer(matching_tweets,filter_words,event_time,reported_time)
    unmatched = unmatched_events(filter_words,matching_tweets)
    #print ("\nUnmatched events", unmatched)

    # remove unmatched events
    df = remove_unmached(df,unmatched)
    # let's see properties of the matched Events
    incident_duration = event_properties (matching_tweets,reported_time,True)

    ''' Life time of a tweet is important to decide on which tweets are relevant,
    from the above code we can see that the average event lasts around 127 minutes (median = 125)
    the max and min are 287 and 26 mins repectively.Since the max duration of an event is 287 mins 
    we can assume that a tweet is not relevant after 287 minutes. This means tweet data will help us
    detect incidents as long as it is not older than 287 mins (4 hours 27 mins). Since a tweet should
    have the same relevance with time, we should encode the tweet age somehow into the input.
    i.e --> new_feature = 287 - tweet age , tweet age = current timestamp - tweet timestamp
    if tweet age is >= 287 then new_feature = 0;
    '''

    # add the new feature as to the dataframe / FUSING THE DATASET
    tweet_feature,binary_feature = add_tweet(df,matching_tweets,lifetime)
    df['tweet_lifetime'] = tweet_feature
    df['tweet_binary'] = binary_feature
    df['tweet_binary']=df['tweet_binary'].astype("category")
    return df

if __name__ == '__main__':
  df = database_creation()   
  print (df.info())

 

