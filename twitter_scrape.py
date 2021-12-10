import tweepy
import csv
import numpy as np
import pandas as pd
import os
from twitter_pos_scrape import pos_scrape
from twitter_neg_scrape import neg_scrape
from twitter_test_scrape import test_scrape


# please change the number of data and lowercase status before scraping
lowercase_data = False
num_pos = 101
num_neg = 101
num_test = 101
print("Positive data:", num_pos)
print("Negative data:", num_neg)
print("Testing data:", num_test)

dev_path = 'twitter_dev_keys.txt'
key_path = './keywords.tsv'
key_test_path = './keywords_test.tsv'
pos_path = './dataset/pos/source/corona_tweets_627.csv'
neg_path = './dataset/neg/info'
test_path = './dataset/test/info'

if not os.path.exists(dev_path):
    print("Twitter API keys not found, please try again!")
    exit()

if not os.path.exists(key_path) or not os.path.exists(key_test_path):
    print("Corona keywords not found, please try again!")
    exit()

if not os.path.exists(pos_path):
    print("Positive dataset source file not found, please try again!")
    exit()

if not os.path.exists(neg_path):
    os.makedirs(neg_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

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

pos_scrape(api, num_pos=num_pos, to_lower=lowercase_data)
neg_scrape(api, num_neg=num_neg, to_lower=lowercase_data)
test_scrape(api, num_test=num_test, to_lower=lowercase_data)
