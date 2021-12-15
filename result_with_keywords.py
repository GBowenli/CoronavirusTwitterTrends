import csv
import numpy as np


def use_keywords():

    # corona keywords in the positive set
    corona_keywords = np.loadtxt('./keywords.tsv', dtype=str, delimiter='\n')

    # write for comparison test
    pos_file = open('./dataset/test/comparison/testset_pos_tweets_pruned_lower.csv', 'r', encoding='utf-8')
    neg_file = open('./dataset/test/comparison/testset_neg_tweets_pruned_lower.csv', 'r', encoding='utf-8')
    pos = pos_file.readlines()
    neg = neg_file.readlines()

    file = open('./dataset/test/comparison/baseline_result.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)

    write_result(pos, neg, corona_keywords, writer)

    # write for general test
    pos_file = open('./dataset/test/general/testset_pos_tweets_pruned_lower.csv', 'r', encoding='utf-8')
    neg_file = open('./dataset/test/general/testset_neg_tweets_pruned_lower.csv', 'r', encoding='utf-8')
    pos = pos_file.readlines()
    neg = neg_file.readlines()

    file = open('./dataset/test/general/baseline_result.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)

    write_result(pos, neg, corona_keywords, writer)


def write_result(pos, neg, corona_keywords, writer):
    for i, text in enumerate(pos):
        if any(keyword in text.lower() for keyword in corona_keywords):
            writer.writerow('1')
        else:
            writer.writerow('0')

    for i, text in enumerate(neg):
        if any(keyword in text.lower() for keyword in corona_keywords):
            writer.writerow('1')
        else:
            writer.writerow('0')
