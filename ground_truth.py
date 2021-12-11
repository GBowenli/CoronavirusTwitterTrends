import numpy


def truth_result(pos_path, neg_path):

    test_pos = numpy.loadtxt(pos_path, delimiter='\n', encoding='utf-8', dtype=str)
    test_neg = numpy.loadtxt(neg_path, delimiter='\n', encoding='utf-8', dtype=str)
    return numpy.concatenate((numpy.ones(test_pos.shape), numpy.zeros(test_neg.shape)), axis=0)
