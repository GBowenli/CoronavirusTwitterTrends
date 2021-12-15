import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Allow memory growth for the GPU
# TODO: remove if not running with GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# character level CNN model
# need to use MaxPool2D with a 1D pool_size because input tensor is 4D (batch_size, vocabulary_size, sentence_size, 1)
class CharacterLevelCNN(Model):
    def __init__(self):
        super(CharacterLevelCNN, self).__init__()
        self.conv1 = layers.Conv1D(256, 7, activation='relu')
        self.pool1 = layers.MaxPool2D((1,3))
        self.conv2 = layers.Conv1D(256, 7, activation='relu')
        self.pool2 = layers.MaxPool2D((1,2))
        self.conv3 = layers.Conv1D(256, 3, activation='relu')
        self.conv4 = layers.Conv1D(256, 3, activation='relu')
        self.conv5 = layers.Conv1D(256, 3, activation='relu')
        self.conv6 = layers.Conv1D(256, 3, activation='relu')
        self.pool6 = layers.MaxPool2D((1,2))
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(1024)
        self.dropout1 = layers.Dropout(0.3)
        self.d2 = layers.Dense(128)
        self.dropout2 = layers.Dropout(0.3)
        self.d3 = layers.Dense(2, activation=tf.keras.activations.softmax)

    def call(self, x):
        # add an extra dimension for filter layer
        # transform 3D tensor to 4D
        x = tf.expand_dims(x, -1)

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
vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789""")

# this method performs one hot encoding to a given text and returns a matrix of size (36, 128)
def transform_text_to_matrix(text):
    num_of_columns = 128

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

  # convert predictions with argmax
  predictions_converted = tf.math.argmax(predictions, axis=1)
  train_accuracy(labels, predictions_converted)

# function to validate the model
@tf.function
def validate_step(text, labels):
  predictions = model(text, training=False)
  t_loss = loss_object(labels, predictions)

  validation_loss(t_loss)

  # convert predictions with argmax
  predictions_converted = tf.math.argmax(predictions, axis=1)
  validation_accuracy(labels, predictions_converted)

# function to test the model
@tf.function
def test_step(text, labels):
  predictions = model(text, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)

  # convert predictions with argmax
  predictions_converted = tf.math.argmax(predictions, axis=1)
  test_accuracy(labels, predictions_converted)
  test_precision(labels, predictions_converted)
  test_recall(labels, predictions_converted)

#*****************************************************************************************
# beginning of script
# get positive and negative datasets
#*****************************************************************************************

# load positive data from
# TODO: change the file location for training pos dataset
pos_dataset_file = open('./dataset/pos/pos_tweets_pruned_lower.csv', 'r', encoding='utf-8')
pos_dataset = pos_dataset_file.readlines()[:30000]

# TODO: change the file location for training neg dataset
neg_dataset_file = open('./dataset/neg/neg_tweets_pruned_lower.csv', 'r', encoding='utf-8')

neg_dataset = neg_dataset_file.readlines()[:30000]

# find split indices (changed to len(pos_dataset) when used for testing new datasets)
pos_split_index = len(pos_dataset)
#pos_split_index = len(pos_dataset) * 4 // 5

neg_split_index = len(neg_dataset)
#neg_split_index = len(neg_dataset) * 4 // 5

# split pos and neg datasets to train, validation sets
pos_train = pos_dataset[:pos_split_index]
pos_validation = pos_dataset[pos_split_index:]

neg_train = neg_dataset[:neg_split_index]
neg_validation = neg_dataset[neg_split_index:]

# combine positive and negative sets to make train and test sets
x_train_text = np.concatenate((pos_train, neg_train))
x_validation_text = np.concatenate((pos_validation, neg_validation))

# create y_train, where 0 is neg, 1 is pos
y_train = np.zeros(x_train_text.size)
y_train[:len(pos_train)] = 1

# create y_validation, where 0 is neg, 1 is pos
y_validation = np.zeros(x_validation_text.size)
y_validation[:len(pos_validation)] = 1

# transform train set with one hot encoding
x_train = []
for index, x_train_sentence in enumerate(x_train_text):
    train_sentence_encoded = transform_text_to_matrix(x_train_sentence)
    x_train.append(train_sentence_encoded)

# trainsform validation set with one hot encoding
x_validation = []
for index, x_validation_sentence in enumerate(x_validation_text):
    validation_sentence_encoded = transform_text_to_matrix(x_validation_sentence)
    x_validation.append(validation_sentence_encoded)

# use tf.data to batch and shuffle dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100000).batch(32)
validation_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(32)

#*****************************************************************************************
# selecting loss, accuracy metrics and optimizer
#*****************************************************************************************

# choose optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# select train and validation loss and accuracy for the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.keras.metrics.Accuracy(name='validation_accuracy')

#*****************************************************************************************
# training and validating CNN model
#*****************************************************************************************

# Create an instance of the model
model = CharacterLevelCNN()
EPOCHS = 20

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  validation_loss.reset_states()
  validation_accuracy.reset_states()

  for train_text, train_labels in train_ds:
    train_step(train_text, train_labels)

  for validation_text, validation_labels in validation_ds:
    validate_step(validation_text, validation_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Validation Loss: {validation_loss.result()}, '
    f'Validation Accuracy: {validation_accuracy.result() * 100}'
  )

#*****************************************************************************************
# testing of new dataset from different date with new keywords
#*****************************************************************************************

print("testing of new dataset from different date with new keywords")

# load test datasets from csv files
# TODO: change the file location for testing pos dataset
pos_test_text_file = open('./dataset/test/general/testset_pos_tweets_pruned_lower.csv', 'r', encoding='utf-8')
pos_test_text = pos_test_text_file.readlines()

# TODO: change the file location for testing neg dataset
neg_test_text_file = open('./dataset/test/general/testset_neg_tweets_pruned_lower.csv', 'r', encoding='utf-8')

neg_test_text = neg_test_text_file.readlines()

# combine positive and negative sets to make new test set
x_new_test_text = np.concatenate((pos_test_text, neg_test_text))

# create y_test, where 0 is neg, 1 is pos
y_new_test = np.zeros(x_new_test_text.size)
y_new_test[:len(pos_test_text)] = 1

# convert test dataset sentences to matrix of size (36, 128)
x_new_test = []
for index, x_new_test_sentence in enumerate(x_new_test_text):
    test_new_sentence_encoded = transform_text_to_matrix(x_new_test_sentence)
    x_new_test.append(test_new_sentence_encoded)

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
test_precision = tf.keras.metrics.Precision(name='test_precision')
test_recall = tf.keras.metrics.Recall(name='test_recall')

test_new_ds = tf.data.Dataset.from_tensor_slices((x_new_test, y_new_test)).batch(32)

test_loss.reset_states()
test_accuracy.reset_states()
test_precision.reset_states()
test_recall.reset_states()

for test_text, test_labels in test_new_ds:
  test_step(test_text, test_labels)
print(
  f'test Loss: {test_loss.result()}, '
  f'test Accuracy: {test_accuracy.result() * 100}, '
  f'test Precision: {test_precision.result() * 100}, '
  f'test Recall: {test_recall.result() * 100}, '
  f'test F1 Score: {2*test_precision.result()*test_recall.result() / (test_precision.result()+test_recall.result())}'
)
