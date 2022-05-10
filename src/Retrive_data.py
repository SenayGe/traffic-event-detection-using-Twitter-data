import pandas as pd
from filter_tweets import database_creation

def main():
    # Retrive data ( dataframe format) 
    data = database_creation(lifetime=278) # You can change lifetime here
    print(data.info())
    print(data['tweet_binary'])
    print(data['tweet_lifetime'])

if __name__ == '__main__':
    main()