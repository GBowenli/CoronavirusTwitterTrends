import csv
import tweepy
import numpy as np
import pandas as pd
from data_preprocessing import preprocessing


def general_scrape(api, num_test=100, trial=200, to_lower=True):
    # corona keywords in the positive set
    corona_keywords = np.loadtxt('./keywords_test.tsv', dtype=str, delimiter='\n')

    df = pd.read_csv('./dataset/test/general/source/corona_tweets_631.csv', header=None, index_col=False,
                     names=["id", "score"],
                     dtype={"id": str, "score": str})

    file_pos = open('./dataset/test/general/testset_pos_tweets.csv', 'w', encoding='utf-8', newline='')
    file_neg = open('./dataset/test/general/testset_neg_tweets.csv', 'w', encoding='utf-8', newline='')
    file_info = open('./dataset/test/general/info/testset_tweets_info.csv', 'w', encoding='utf-8', newline='')
    writer_pos = csv.writer(file_pos)
    writer_neg = csv.writer(file_neg)
    writer_info = csv.writer(file_info)

    length_of_tweets = df["id"].shape[0]

    current_twitter_ids = []

    count = 0
    for index, tweet_id in enumerate(df["id"]):
        if count == num_test:
            break

        current_twitter_ids.extend([tweet_id])

        if index % 100 == 0:
            statuses = api.lookup_statuses(current_twitter_ids)
            current_twitter_ids = []

        elif index == length_of_tweets - 1 and len(current_twitter_ids) > 0:
            statuses = api.lookup_statuses(current_twitter_ids)

        else:
            statuses = []

        if (index % 100 == 0) or (index == length_of_tweets - 1 and len(current_twitter_ids) > 0):
            for status in statuses:
                # replace newline characters with space
                if (not status.retweeted) and ('RT @' not in status.text):
                    tweet_text = status.text.replace('\n', ' ')

                    if any(keyword in tweet_text.lower() for keyword in corona_keywords):
                        writer_pos.writerow([tweet_text])
                        writer_info.writerow([tweet_text, 1, status.id, status.created_at, status.geo])
                        count += 1
                        print(count)
                        if count == num_test:
                            break

    # search keyword 'e' as it is the most common letter
    search_words = "e -filter:retweets"

    # change the number of items to 60000 to get the negative set
    tweets = tweepy.Cursor(api.search_tweets,
                           q=search_words,
                           lang="en").items(trial)

    count = 0
    for index, tweet in enumerate(tweets):
        # replace newline characters with space
        tweet_text = tweet.text.replace('\n', ' ')
        if not any(keyword in tweet_text.lower() for keyword in corona_keywords):
            writer_neg.writerow([tweet_text])
            writer_info.writerow([tweet_text, 0, tweet.id, tweet.created_at, tweet.geo])
            count += 1
            if count == num_test:
                break

    file_pos.close()
    file_neg.close()
    file_info.close()

    if to_lower:
        preprocessing('./dataset/test/general/testset_pos_tweets.csv',
                      './dataset/test/general/testset_pos_tweets_pruned_lower.csv',
                      to_lower=to_lower)
        preprocessing('./dataset/test/general/testset_neg_tweets.csv',
                      './dataset/test/general/testset_neg_tweets_pruned_lower.csv',
                      to_lower=to_lower)
    else:
        preprocessing('./dataset/test/general/testset_pos_tweets.csv',
                      './dataset/test/general/testset_pos_tweets_pruned_normal.csv',
                      to_lower=to_lower)
        preprocessing('./dataset/test/general/testset_neg_tweets.csv',
                      './dataset/test/general/testset_neg_tweets_pruned_normal.csv',
                      to_lower=to_lower)
