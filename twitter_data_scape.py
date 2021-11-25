import tweepy
import numpy as np
import csv

# load txt file that contains twitter dev keys
dev_keys = np.loadtxt("twitter_dev_keys.txt", dtype=str, delimiter="\n")

# set twitter dev keys in variables
consumer_key = dev_keys[0]
consumer_secret = dev_keys[1]
access_token = dev_keys[2]
access_token_secret = dev_keys[3]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# load txt file that contains coronavirus twitter ids
corona_iota_tweet_ids = np.loadtxt('twitter_id_pos/corona_tweets_iota.csv', dtype=str, delimiter='\n')

# open file in write mode
file = open('scraped_tweets_pos/iota_tweets.csv', 'w', encoding="utf-8")
writer = csv.writer(file)

for tweet_id in corona_iota_tweet_ids:
    tweets = api.lookup_statuses([tweet_id])

    for tweet in tweets:
        print('got tweet')
        writer.writerow(tweet.text)

    break

file.close()