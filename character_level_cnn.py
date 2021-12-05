import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# character level CNN model
# need to use MaxPool2D with a 1D pool_size because input tensor is 4D (batch_size, vocabulary_size, sentence_size, 1)
class CharacterLevelCNN(Model):
    def __init__(self):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = layers.Conv1D(256, 7, activation='relu', input_shape=(None,62,256,1))
        self.pool1 = layers.MaxPool2D((1,3))
        self.conv2 = layers.Conv1D(256, 7, activation='relu')
        self.pool2 = layers.MaxPool2D((1,3))
        self.conv3 = layers.Conv1D(256, 3, activation='relu')
        self.conv4 = layers.Conv1D(256, 3, activation='relu')
        self.conv5 = layers.Conv1D(256, 3, activation='relu')
        self.conv6 = layers.Conv1D(256, 3, activation='relu')
        self.pool6 = layers.MaxPool2D((1,3))
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(1024)
        self.dropout1 = layers.Dropout(0.5)
        self.d2 = layers.Dense(1024)
        self.dropout2 = layers.Dropout(0.5)
        self.d3 = layers.Dense(2)

    def call(self, x):
        # add an extra dimension for filter layer
        # transform 3D tensor to 4D
        x = tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2], 1])

        print("init size: " + str(x.get_shape()))

        x = self.conv1(x)

        print("conv1 size: " + str(x.get_shape()))

        x = self.pool1(x)

        print("pool1 size: " + str(x.get_shape()))

        x = self.conv2(x)

        print("conv2 size: " + str(x.get_shape()))

        x = self.pool2(x)

        print("pool2 size: " + str(x.get_shape()))

        x = self.conv3(x)

        print("conv3 size: " + str(x.get_shape()))

        x = self.conv4(x)

        print("conv4 size: " + str(x.get_shape()))

        x = self.conv5(x)

        print("conv5 size: " + str(x.get_shape()))

        x = self.conv6(x)

        print("conv6 size: " + str(x.get_shape()))

        x = self.pool6(x)

        print("pool6 size: " + str(x.get_shape()))

        x = self.flatten(x)

        print("flatten size: " + str(x.get_shape()))

        x = self.d1(x)

        print("d1 size: " + str(x.get_shape()))

        x = self.dropout1(x)
        x = self.d2(x)

        print("d2 size: " + str(x.get_shape()))

        x = self.dropout2(x)
        x = self.d3(x)

        print("d3 size: " + str(x.get_shape()))

        return x

# define all accepted characters
vocabulary = list("""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789""")

# this method performs one hot encoding to a given text and returns a matrix of size (62, 256)
def transform_text_to_matrix(text):
    num_of_columns = 256

    matrix = np.zeros((len(vocabulary), num_of_columns))

    for index, char in enumerate(text):
        if char != ' ' and char != '\n':
            pos_in_vocabulary = vocabulary.index(char)
            matrix[pos_in_vocabulary][index] = 1

        if index == num_of_columns-1:
            break

    return matrix

# function to train the model with tf.GradientTape
@tf.function
def train_step(text, labels):
  with tf.GradientTape() as tape:
    predictions = model(text, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

# function to test the model
@tf.function
def test_step(text, labels):
  predictions = model(text, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

#*****************************************************************************************
# beginning of script
# get positive and negative datasets
#*****************************************************************************************

# load positive data from csv
pos_dataset_file = open('scraped_tweets_pos/iota_tweets_pruned.csv', 'r', encoding='utf-8')
#TODO: remove the [:100] used for testing!
pos_dataset = pos_dataset_file.readlines()[:100]

print(len(pos_dataset))

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
x_train_text = np.concatenate((pos_train, neg_train))
x_test_text = np.concatenate((pos_test, neg_test))

# create y_train, where 0 is neg, 1 is pos
y_train = np.zeros(x_train_text.size)
y_train[:len(pos_train)] = 1

# create y_test, where 0 is neg, 1 is pos
y_test = np.zeros(x_test_text.size)
y_test[:len(pos_test)] = 1

# transform train set with one hot encoding
x_train = []
for index, x_train_sentence in enumerate(x_train_text):
    train_sentence_encoded = transform_text_to_matrix(x_train_sentence)
    x_train.append(train_sentence_encoded)

# transform test set with one hot encoding
x_test = []
for index, x_test_sentence in enumerate(x_test_text):
    test_sentence_encoded = transform_text_to_matrix(x_test_sentence)
    x_test.append(test_sentence_encoded)

# use tf.data to batch and shuffle dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#*****************************************************************************************
# selecting loss, accuracy metrics and optimzer
#*****************************************************************************************

# choose optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# select train and test loss and accuracy for the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#*****************************************************************************************
# training CNN model
#*****************************************************************************************

# Create an instance of the model
model = CharacterLevelCNN()
EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for text, labels in train_ds:
    train_step(text, labels)

  for test_text, test_labels in test_ds:
    test_step(test_text, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )