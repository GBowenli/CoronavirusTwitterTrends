import tensorflow as tf
from tensorflow.keras import layers, Model

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
        self.d2 = layers.Dense(1024)
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
        x = self.d2(x)
        x = self.d3(x)
        return x

model = CharacterLevelCNN()

model.build((62, 500, 1))

model.summary()