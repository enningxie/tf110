# explore overfitting and underfitting
# To prevent overfitting, the best solution is to use more training data.
# the next best solution is to use techniques like regularization.
# two common regularization techniques weight regularization and dropout.
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# download the IMDB dataset
NUM_WORDS = 10000

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of result[i] to 1
    return results


train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data = multi_hot_sequences(test_data, NUM_WORDS)

# demonstrate overfitting
# The simplest way to prevent overfitting is to reduce the size of the model.
# Always keep this in mind: deep learning models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting.

# create a baseline model
baseline_model = keras.Sequential([
    # input_shape is only required here so that `.summary` works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy]
)

baseline_model.summary()

print('baseline_model start.')

baseline_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=1
)

# create a smaller model
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy]
)

smaller_model.summary()

print('smaller_model start.')

smaller_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=1
)

# create a bigger model
bigger_model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy]
)

bigger_model.summary()

print('bigger_model start.')

bigger_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=1
)

# remember: a lower validation loss indicates a better model
# simpler models are less likely to overfit than complex ones.

# add weight regularization
l2_model = keras.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                 loss=keras.losses.binary_crossentropy,
                 metrics=[keras.metrics.binary_accuracy])

print('l2_model start.')

l2_model.fit(train_data, train_labels,
             epochs=20,
             batch_size=512,
             validation_data=(test_data, test_labels),
             verbose=1)

# add dropout
dpt_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy]
)

print('dpt_model start.')

dpt_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=1
)


# conclusion:
# here the most common ways to prevent overfitting in neural networks:
# 1. Get more training data.
# 2. Reduce the capacity of the network.
# 3. Add weight regularization.
# 4. Add dropout.
# 5. data-augmentation and batch normalization.