import csv
import tweepy
import numpy as np
import pandas as pd
from data_preprocessing import preprocessing


def pos_scrape(api, num_pos=100, to_lower=True):

    # load txt file that contains coronavirus twitter ids
    # *** simply replace the *.csv path to a different csv to data scrape from a different twitter id file
    df = pd.read_csv('./dataset/pos/source/corona_tweets_627.csv', header=None, index_col=False, names=["id", "score"],
                     dtype={"id": str, "score": str})

    # open file in write mode
    # *** the following directory and file need to be created in order for the code to function properly
    file = open('./dataset/pos/pos_tweets.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)

    # total number of tweets to look up
    # length_of_tweets = corona_iota_tweet_ids.shape[0]
    length_of_tweets = df["id"].shape[0]

    current_twitter_ids = []

    # loop through every twee id, every 100 tweets
    # we call lookup_statuses with the current 100 tweets to get their statuses
    # we write the text of the every status in a csv file
    # if index reaches end of the end, if current_twitter_ids is not empty, get their statuses as well
    # for index, tweet_id in enumerate(corona_iota_tweet_ids):
    for index, tweet_id in enumerate(df["id"]):
        if index == num_pos+1:
            break

        print(index)

        current_twitter_ids.extend([tweet_id])

        if index % 100 == 0:
            statuses = api.lookup_statuses(current_twitter_ids)

            current_twitter_ids = []

            for status in statuses:
                # replace newline characters with space
                if (not status.retweeted) and ('RT @' not in status.text):
                    tweet_text = status.text.replace('\n', ' ')

                    writer.writerow([tweet_text])

        elif index == length_of_tweets - 1:
            if len(current_twitter_ids) > 0:
                statuses = api.lookup_statuses(current_twitter_ids)

                for status in statuses:
                    # replace newline characters with space
                    if (not status.retweeted) and ('RT @' not in status.text):
                        tweet_text = status.text.replace('\n', ' ')

                        writer.writerow([tweet_text])

    # close file
    file.close()

    if to_lower:
        preprocessing('./dataset/pos/pos_tweets.csv', './dataset/pos/pos_tweets_pruned_lower.csv', to_lower=to_lower)
    else:
        preprocessing('./dataset/pos/pos_tweets.csv', './dataset/pos/pos_tweets_pruned_normal.csv', to_lower=to_lower)
