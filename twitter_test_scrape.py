import csv
import tweepy
import numpy as np
from data_preprocessing import preprocessing


def test_scrape(api, num_test=101, to_lower=True):
    # search keyword 'e' as it is the most common letter
    search_words = "e -filter:retweets"

    # change the number of items to 60000 to get the negative set
    tweets = tweepy.Cursor(api.search_tweets,
                           q=search_words,
                           lang="en").items(num_test)

    # corona keywords in the positive set
    corona_keywords = np.loadtxt('./keywords_test.tsv', dtype=str, delimiter='\n')

    file_pos = open('./dataset/test/testset_pos_tweets.csv', 'w', encoding='utf-8', newline='')
    file_neg = open('./dataset/test/testset_neg_tweets.csv', 'w', encoding='utf-8', newline='')
    file_info = open('./dataset/test/info/testset_tweets_info.csv', 'w', encoding='utf-8', newline='')

    writer_pos = csv.writer(file_pos)
    writer_neg = csv.writer(file_neg)
    writer_info = csv.writer(file_info)

    for index, tweet in enumerate(tweets):
        # replace newline characters with space
        tweet_text = tweet.text.replace('\n', ' ')
        if any(keyword in tweet_text.lower() for keyword in corona_keywords):
            writer_pos.writerow([tweet_text])
            writer_info.writerow([tweet_text, 1, tweet.id, tweet.created_at, tweet.geo])

        else:
            writer_neg.writerow([tweet_text])
            writer_info.writerow([tweet_text, 0, tweet.id, tweet.created_at, tweet.geo])

    file_pos.close()
    file_neg.close()
    file_info.close()

    if to_lower:
        preprocessing('./dataset/test/testset_pos_tweets.csv', './dataset/test/testset_pos_tweets_pruned_lower.csv',
                      to_lower=to_lower)
        preprocessing('./dataset/test/testset_neg_tweets.csv', './dataset/test/testset_neg_tweets_pruned_lower.csv',
                      to_lower=to_lower)
    else:
        preprocessing('./dataset/test/testset_pos_tweets.csv', './dataset/test/testset_pos_tweets_pruned_normal.csv',
                      to_lower=to_lower)
        preprocessing('./dataset/test/testset_neg_tweets.csv', './dataset/test/testset_neg_tweets_pruned_normal.csv',
                      to_lower=to_lower)
