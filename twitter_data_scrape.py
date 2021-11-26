import tweepy
import numpy as np
import csv

# load txt file that contains twitter dev keys
dev_keys = np.loadtxt('twitter_dev_keys.txt', dtype=str, delimiter='\n')

# set twitter dev keys in variables
consumer_key = dev_keys[0]
consumer_secret = dev_keys[1]
access_token = dev_keys[2]
access_token_secret = dev_keys[3]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# load txt file that contains coronavirus twitter ids
# *** simply replace the *.csv path to a different csv to data scrape from a different twitter id file
corona_iota_tweet_ids = np.loadtxt('twitter_id_pos/corona_tweets_iota.csv', dtype=str, delimiter='\n')

# open file in write mode
# *** the following directory and file need to be created in order for the code to function properly
file = open('scraped_tweets_pos/iota_tweets.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(file)

# total number of tweets to look up
length_of_tweets = corona_iota_tweet_ids.shape[0]

current_twitter_ids = []

# loop through every twee id, every 100 tweets, we call lookup_statuses with the current 100 tweets to get their statuses
# we write the text of the every status in a csv file
# if index reaches end of the end, if current_twitter_ids is not empty, get their statuses as well
for index, tweet_id in enumerate(corona_iota_tweet_ids):
    print(index)
    
    current_twitter_ids.extend([tweet_id])

    if index % 100 == 0:
        statuses = api.lookup_statuses(current_twitter_ids)

        current_twitter_ids = []

        for status in statuses:
            # replace newline characters with space
            tweet_text = status.text.replace('\n', ' ')

            writer.writerow([tweet_text])

    elif index == length_of_tweets-1:
       if len(current_twitter_ids) > 0:
            statuses = api.lookup_statuses(current_twitter_ids)

            for status in statuses:
                # replace newline characters with space
                tweet_text = status.text.replace('\n', ' ')

                writer.writerow([tweet_text])

# close file
file.close()