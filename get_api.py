import numpy as np
import tweepy


def get_api():
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

    return api
