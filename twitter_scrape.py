import os
from get_api import get_api
from twitter_pos_scrape import pos_scrape
from twitter_neg_scrape import neg_scrape
from general_test_scrape import general_scrape
from comparison_test_scrape import compare_scrape
from result_with_keywords import use_keywords
from baseline import baseline_result


# please change the number of data and lowercase status before scraping
lowercase_data = True
num_pos = 60000
num_neg = 60000
num_test = 6000
print("Positive data:", num_pos)
print("Negative data:", num_neg)
print("Testing data:", num_test)

dev_path = 'twitter_dev_keys.txt'
key_path = './keywords.tsv'
key_test_path = './keywords_test.tsv'
pos_path = './dataset/pos/source/corona_tweets_627.csv'
neg_path = './dataset/neg/info'
general_test_source = './dataset/test/general/source/corona_tweets_631.csv'
general_test_path = './dataset/test/general/info'
comparison_test_path = './dataset/test/comparison/info'
model_path = './models'
results_path = './dataset/test/results'

if not os.path.exists(dev_path):
    print("Twitter API keys not found, please try again!")
    exit()

if not os.path.exists(key_path) or not os.path.exists(key_test_path):
    print("Corona keywords not found, please try again!")
    exit()

if not os.path.exists(pos_path) or not os.path.exists(general_test_source):
    print("Dataset source file not found, please try again!")
    exit()

if not os.path.exists(neg_path):
    os.makedirs(neg_path)

if not os.path.exists(general_test_path):
    os.makedirs(general_test_path)

if not os.path.exists(comparison_test_path):
    os.makedirs(comparison_test_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(results_path):
    os.makedirs(results_path)

api = get_api()

# # scraping starts here:
pos_scrape(api, num_pos=num_pos, to_lower=lowercase_data)
neg_scrape(api, num_neg=num_neg, to_lower=lowercase_data)
general_scrape(api, num_test=num_test, trial=2*num_test, to_lower=lowercase_data)
compare_scrape(api, num_test=num_test, trial=2*num_test, to_lower=lowercase_data)

# generate result (baseline_result.csv) using keywords for testing
use_keywords()

# print baseline evaluation results
baseline_result()
