from datetime import datetime
import tweepy
import csv 

timestamp = 1485893698.0

dt = datetime.fromtimestamp(timestamp)

print (dt)


api_key = "LZ5N8bF2RWGOLuaQ6n8shjLon"
api_key_secret = "urhvy4gt9n9PbA3cORO2GXyFh2crka4qAX9vhkxScPZU1xMPel"
access_token= "1242795460009562118-hAgKdoc08XEnKofJGEEOsmKz9Jw3EK"
access_token_secret = "gdMkOOFbqzvATVnyEQDlkUz308qj1TYoYrD3NQLiPOnRW"

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

tweetFetched = api.get_status(826553825234403328)
print("Tweet fetched" + tweetFetched.text)

