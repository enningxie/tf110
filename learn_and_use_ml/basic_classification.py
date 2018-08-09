# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras


# helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# import the Fasion MNIST dataset
fasion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()

print(len(train_images), len(test_images))
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.show()

# preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['acc']
)

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss: {}, test_acc: {}.'.format(test_loss, test_acc))

# make predictions
predictions = model.predict(test_images)

assert np.argmax(predictions[0]) == test_labels[0]

img = test_images[0]
print(img.shape)

# add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)

predictions_ = model.predict(img)

assert np.argmax(predictions_[0]) == test_labels[0], 'lol'
