import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# character level CNN model
class CharacterLevelCNN(Model):
    def __init__(self):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = layers.Conv1D(256, 7, activation='relu')
        self.pool1 = layers.MaxPool1D(3)
        self.conv2 = layers.Conv1D(256, 7, activation='relu')
        self.pool2 = layers.MaxPool1D(3)
        self.conv3 = layers.Conv1D(256, 3, activation='relu')
        self.conv4 = layers.Conv1D(256, 3, activation='relu')
        self.conv5 = layers.Conv1D(256, 3, activation='relu')
        self.conv6 = layers.Conv1D(256, 3, activation='relu')
        self.pool6 = layers.MaxPool1D(3)
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(1024)
        self.dropout1 = layers.Dropout(0.5)
        self.d2 = layers.Dense(1024)
        self.dropout2 = layers.Dropout(0.5)
        self.d3 = layers.Dense(2)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool6(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.dropout2(x)
        x = self.d3(x)
        return x

# define all accepted characters
vocabulary = list("""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789""")

# this method performs one hot encoding to a given text and returns a matrix of size (62, 256, 1)
def transform_text_to_matrix(text):
    num_of_columns = 256

    matrix = np.zeros((len(vocabulary), num_of_columns, 1))

    for index, char in enumerate(text):
        if char != ' ' and char != '\n':
            pos_in_vocabulary = vocabulary.index(char)
            matrix[pos_in_vocabulary][index][0] = 1

        if index == num_of_columns-1:
            break

    return matrix

#*****************************************************************************************
# beginning of script
# get positive and negative datasets
#*****************************************************************************************

# load positive data from csv
pos_dataset_file = open('scraped_tweets_pos/iota_tweets_pruned.csv', 'r', encoding='utf-8')
pos_dataset = pos_dataset_file.readlines()

# TODO: load negative data from csv
neg_dataset = []

# find split indices
pos_split_index = len(pos_dataset) * 7 // 10
neg_split_index = len(neg_dataset) * 7 // 10

# split pos and neg datasets to train and test sets
pos_train = pos_dataset[:pos_split_index]
pos_test = pos_dataset[pos_split_index:]
neg_train = neg_dataset[:neg_split_index]
neg_test = neg_dataset[neg_split_index:]

# combine positive and negative sets to make train and test sets
train_set = np.concatenate((pos_train, neg_train))
test_set = np.concatenate((pos_test, neg_test))

# create y_train, where 0 is neg, 1 is pos
y_train = np.zeros(train_set.size)
y_train[:len(pos_train)] = 1

# create y_test, where 0 is neg, 1 is pos
y_test = np.zeros(test_set.size)
y_test[:len(pos_test)] = 1

# transform train and test sets with one hot encoding
train_set_encoded = []

for index, train_sentence in enumerate(train_set):
    train_sentence_matrix = transform_text_to_matrix(train_sentence)
    #train_set_encoded.extend(train_sentence_matrix)

print(len(train_set_encoded))

# matrix = transform_text_to_matrix("ab ab ab")
# print(matrix.shape)

# model = CharacterLevelCNN()
# model.build((62, 256, 1))
# model.summary()