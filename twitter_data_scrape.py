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
file = open('scraped_tweets_pos/iota_tweets.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(file)

length_of_tweets = corona_iota_tweet_ids.shape[0]

current_twitter_ids = []

for index, tweet_id in enumerate(corona_iota_tweet_ids):
    print(index)
    
    current_twitter_ids.extend([tweet_id])

    if index % 100 == 0:
        statuses = api.lookup_statuses(current_twitter_ids)

        current_twitter_ids = []

        for status in statuses:
            tweet_text = status.text.replace("\n", " ")

            writer.writerow([tweet_text])

   elif index == length_of_tweets-1:
       if len(current_twitter_ids) > 0:
            statuses = api.lookup_statuses(current_twitter_ids)

            for status in statuses:
                tweet_text = status.text.replace("\n", " ")

                writer.writerow([tweet_text])

file.close()