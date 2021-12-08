import numpy

from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

df_pos = numpy.loadtxt('./dataset/pos/pos_tweets_pruned_normal.csv', delimiter='\n', encoding='utf-8', dtype=str)
df_neg = numpy.loadtxt('./dataset/neg/neg_tweets_pruned_normal.csv', delimiter='\n', encoding='utf-8', dtype=str)
twitter_text = numpy.concatenate((df_pos[0:50000], df_neg[0:50000]), axis=0)
twitter_cat = numpy.concatenate((numpy.ones(50000), numpy.zeros(50000)), axis=0)


# Normalize data after vectorization process is finished
def nomalize_data(train, test):
    normalizer_train = Normalizer().fit(X=train)
    train = normalizer_train.transform(train)
    test = normalizer_train.transform(test)
    return train, test


# Comparison: count vector without any other specification
def count_vector(train_set, test_set):
    vectorizer = CountVectorizer()
    train = vectorizer.fit_transform(train_set)
    test = vectorizer.transform(test_set)
    return nomalize_data(train, test)


# Comparison: tfidf vector without any other specification
def tfidf_vector(train_set, test_set):
    vectorizer = TfidfVectorizer()
    train = vectorizer.fit_transform(train_set)
    test = vectorizer.transform(test_set)
    return nomalize_data(train, test)


# First data-preprocessing method: count vector with stopwords
def count_vec_with_sw(train_set, test_set):
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(stop_words=stop_words)
    train = vectorizer.fit_transform(train_set)
    test = vectorizer.transform(test_set)
    return nomalize_data(train, test)


# Second data-preprocessing method: use ngram
# in this case, unigram and bigram were used to perform the vectorization.
def count_vec_with_ngram(train_set, test_set):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    train = vectorizer.fit_transform(train_set)
    test = vectorizer.transform(test_set)
    return nomalize_data(train, test)


# Third data-preprocessing method: stem
class StemTokenizer:
    def __init__(self):
        self.wnl = PorterStemmer()

    def __call__(self, doc):
        return [self.wnl.stem(t) for t in word_tokenize(doc) if t.isalpha()]


def count_vec_stem(train_set, test_set):
    vectorizer = CountVectorizer(tokenizer=StemTokenizer())
    train = vectorizer.fit_transform(train_set)
    test = vectorizer.transform(test_set)
    return nomalize_data(train, test)


# Third data-preprocessing method: lemma
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t, pos=get_wordnet_pos(t)) for t in word_tokenize(doc) if t.isalpha()]


def count_vec_lemma(train_set, test_set):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
    train = vectorizer.fit_transform(train_set)
    test = vectorizer.transform(test_set)
    return nomalize_data(train, test)


# ------------------------------ training ------------------------------
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(twitter_text):
    vectors_train, vectors_test = count_vector(twitter_text[train_index], twitter_text[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat[train_index])
    accuracies.append(metrics.accuracy_score(twitter_cat[test_index], clf.predict(vectors_test)))

print("The accuracy of count vectorizer without any pre-processing is: " + str(numpy.mean(accuracies)))

accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(twitter_text):
    vectors_train, vectors_test = tfidf_vector(twitter_text[train_index], twitter_text[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat[train_index])
    accuracies.append(metrics.accuracy_score(twitter_cat[test_index], clf.predict(vectors_test)))

print("The accuracy of tfidf vectorizer without any pre-processing is: " + str(numpy.mean(accuracies)))

accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(twitter_text):
    vectors_train, vectors_test = count_vec_with_sw(twitter_text[train_index], twitter_text[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat[train_index])
    accuracies.append(metrics.accuracy_score(twitter_cat[test_index], clf.predict(vectors_test)))

print("The accuracy of applying stopwords filter in the vectorization process is: " + str(numpy.mean(accuracies)))

accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(twitter_text):
    vectors_train, vectors_test = count_vec_with_ngram(twitter_text[train_index], twitter_text[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat[train_index])
    accuracies.append(metrics.accuracy_score(twitter_cat[test_index], clf.predict(vectors_test)))

print("The accuracy of using unigram and bigram to do the vectorization process is: " + str(numpy.mean(accuracies)))

accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(twitter_text):
    vectors_train, vectors_test = count_vec_stem(twitter_text[train_index], twitter_text[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat[train_index])
    accuracies.append(metrics.accuracy_score(twitter_cat[test_index], clf.predict(vectors_test)))

print("The accuracy of applying stem in the vectorization process is: " + str(numpy.mean(accuracies)))

accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(twitter_text):
    vectors_train, vectors_test = count_vec_lemma(twitter_text[train_index], twitter_text[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat[train_index])
    accuracies.append(metrics.accuracy_score(twitter_cat[test_index], clf.predict(vectors_test)))

print("The accuracy of applying lemma in the vectorization process is: " + str(numpy.mean(accuracies)))
