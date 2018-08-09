# predict house prices: regression
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# the boston housing prices dataset
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle the trianing set. xz
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# examples and features
print('Training set: {}.'.format(train_data.shape))  # 404 examples, 13 features
print('Testing set: {}.'.format(test_data.shape))  # 102 examples, 13 features

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

# Labels
print(train_labels[:10])

# before normalization
print(train_data[0])

# Normalize features
# test data is not used when caculating the mean and std.
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# after normalization
print(train_data[0])


# create the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_absolute_error]
    )
    return model


model = build_model()
model.summary()


# train the model
# display training progress by printing a single dot for each complated epoch.
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 500

# store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[PrintDot()])

# print(history.history.keys())


def plot_history(history_):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.plot(history_.epoch, np.array(history_.history['mean_absolute_error']), label='Train loss')
    plt.plot(history_.epoch, np.array(history_.history['val_mean_absolute_error']), label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)

# conclusion
# 1. Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
# 2. Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
# 3. When input data features have values with different ranges, each feature should be scaled independently.
# 4. If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
# 5. Early stopping is a useful technique to prevent overfitting.