import numpy
from sklearn import metrics
from ground_truth import truth_result


def baseline_result():
    gen_pos = './dataset/test/general/testset_pos_tweets_pruned_lower.csv'
    gen_neg = './dataset/test/general/testset_neg_tweets_pruned_lower.csv'
    com_pos = './dataset/test/comparison/testset_pos_tweets_pruned_lower.csv'
    com_neg = './dataset/test/comparison/testset_neg_tweets_pruned_lower.csv'
    gen_base = './dataset/test/general/baseline_result.csv'
    com_base = './dataset/test/comparison/baseline_result.csv'

    gen_truth = truth_result(gen_pos, gen_neg)
    com_truth = truth_result(com_pos, com_neg)

    gen = numpy.loadtxt(gen_base, delimiter='\n', encoding='utf-8', dtype=int)
    com = numpy.loadtxt(com_base, delimiter='\n', encoding='utf-8', dtype=int)

    print("============================================================")

    # print general testing results
    print("The baseline result for general testing is:")
    print(metrics.accuracy_score(gen_truth, gen))
    print(metrics.f1_score(gen_truth, gen))
    print(metrics.recall_score(gen_truth, gen))
    print(metrics.precision_score(gen_truth, gen))

    print("------------------------------------------------------------")

    # print general testing results
    print("The baseline result for comparison testing result is:")
    print(metrics.accuracy_score(com_truth, com))
    print(metrics.f1_score(com_truth, com))
    print(metrics.recall_score(com_truth, com))
    print(metrics.precision_score(com_truth, com))
