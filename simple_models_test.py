import numpy
import pandas as pd
from joblib import dump, load

from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from sklearn.neural_network import MLPClassifier

import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

df_pos = numpy.loadtxt('./dataset/pos/pos_tweets_pruned_lower.csv', delimiter='\n', encoding='utf-8', dtype=str)
df_neg = numpy.loadtxt('./dataset/neg/neg_tweets_pruned_lower.csv', delimiter='\n', encoding='utf-8', dtype=str)
pos_set = numpy.random.choice(df_pos, size=30000, replace=False)
neg_set = numpy.random.choice(df_neg, size=30000, replace=False)
twitter_text = numpy.concatenate((pos_set, neg_set), axis=0)
twitter_cat = numpy.concatenate((numpy.ones(30000), numpy.zeros(30000)), axis=0)


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


# ------------------------------ testing ------------------------------

# ****************************** general ******************************

test_pos = numpy.loadtxt('./dataset/test/general/testset_pos_tweets_pruned_lower.csv', delimiter='\n', encoding='utf-8', dtype=str)
test_neg = numpy.loadtxt('./dataset/test/general/testset_neg_tweets_pruned_lower.csv', delimiter='\n', encoding='utf-8', dtype=str)
twitter_text_test = numpy.concatenate((test_pos, test_neg), axis=0)
twitter_cat_test = numpy.concatenate((numpy.ones(test_pos.shape), numpy.zeros(test_neg.shape)), axis=0)

vectors_train, vectors_test = count_vec_stem(twitter_text, twitter_text_test)

print("Testing with the general testing dataset:")

# ============================== Logistic Regression ==============================
clf = LogisticRegression(C=40.0, max_iter=1000, random_state=0).fit(vectors_train, twitter_cat)

y_test = clf.predict(vectors_test)

print("----------------------------------------")
print("Logistic Regression: General Test")
print(metrics.accuracy_score(twitter_cat_test, y_test))
print(metrics.f1_score(twitter_cat_test, y_test))
print(metrics.recall_score(twitter_cat_test, y_test))
print(metrics.precision_score(twitter_cat_test, y_test))

df_export = pd.DataFrame(data=y_test)

# export to csv
df_export.to_csv('./dataset/test/results/gen_result_lr.csv', index=False)
dump(clf, './models/lr_gen.npy')

# ============================== SVM ==============================
clf = svm.SVC(kernel='linear', gamma='auto', C=1).fit(vectors_train, twitter_cat)

y_test = clf.predict(vectors_test)

print("----------------------------------------")
print("SVM: General Test")
print(metrics.accuracy_score(twitter_cat_test, y_test))
print(metrics.f1_score(twitter_cat_test, y_test))
print(metrics.recall_score(twitter_cat_test, y_test))
print(metrics.precision_score(twitter_cat_test, y_test))

df_export = pd.DataFrame(data=y_test)

# export to csv
df_export.to_csv('./dataset/test/results/gen_result_svm.csv', index=False)
dump(clf, './models/svm_gen.npy')

# ============================== MLP ==============================
clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat)

y_test = clf.predict(vectors_test)

print("----------------------------------------")
print("MLP: General Test")
print(metrics.accuracy_score(twitter_cat_test, y_test))
print(metrics.f1_score(twitter_cat_test, y_test))
print(metrics.recall_score(twitter_cat_test, y_test))
print(metrics.precision_score(twitter_cat_test, y_test))

df_export = pd.DataFrame(data=y_test)

# export to csv
df_export.to_csv('./dataset/test/results/gen_result_mlp.csv', index=False)
dump(clf, './models/mlp_gen.npy')

# ****************************** comparison ******************************

test_pos = numpy.loadtxt('./dataset/test/comparison/testset_pos_tweets_pruned_lower.csv', delimiter='\n', encoding='utf-8', dtype=str)
test_neg = numpy.loadtxt('./dataset/test/comparison/testset_neg_tweets_pruned_lower.csv', delimiter='\n', encoding='utf-8', dtype=str)
twitter_text_test = numpy.concatenate((test_pos, test_neg), axis=0)
twitter_cat_test = numpy.concatenate((numpy.ones(test_pos.shape), numpy.zeros(test_neg.shape)), axis=0)

vectors_train, vectors_test = count_vec_stem(twitter_text, twitter_text_test)
print("========================================")
print('Testing with the "omicron" testing dataset:')


# ============================== Logistic Regression ==============================
clf = LogisticRegression(C=40.0, max_iter=1000, random_state=0).fit(vectors_train, twitter_cat)

y_test = clf.predict(vectors_test)

print("Logistic Regression: Comparison Test")
print(metrics.accuracy_score(twitter_cat_test, y_test))
print(metrics.f1_score(twitter_cat_test, y_test))
print(metrics.recall_score(twitter_cat_test, y_test))
print(metrics.precision_score(twitter_cat_test, y_test))

df_export = pd.DataFrame(data=y_test)

# export to csv
df_export.to_csv('./dataset/test/results/com_result_lr.csv', index=False)
dump(clf, './models/lr_com.npy')

# ============================== SVM ==============================
clf = svm.SVC(kernel='linear', gamma='auto', C=1).fit(vectors_train, twitter_cat)

y_test = clf.predict(vectors_test)

print("----------------------------------------")
print("SVM: Comparison Test")
print(metrics.accuracy_score(twitter_cat_test, y_test))
print(metrics.f1_score(twitter_cat_test, y_test))
print(metrics.recall_score(twitter_cat_test, y_test))
print(metrics.precision_score(twitter_cat_test, y_test))

df_export = pd.DataFrame(data=y_test)

# export to csv
df_export.to_csv('./dataset/test/results/com_result_svm.csv', index=False)
dump(clf, './models/svm_com.npy')

# ============================== MLP ==============================
clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, twitter_cat)

y_test = clf.predict(vectors_test)

print("----------------------------------------")
print("MLP: Comparison Test")
print(metrics.accuracy_score(twitter_cat_test, y_test))
print(metrics.f1_score(twitter_cat_test, y_test))
print(metrics.recall_score(twitter_cat_test, y_test))
print(metrics.precision_score(twitter_cat_test, y_test))

df_export = pd.DataFrame(data=y_test)

# export to csv
df_export.to_csv('./dataset/test/results/com_result_mlp.csv', index=False)
dump(clf, './models/mlp_com.npy')
