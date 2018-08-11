# this guide uses ml to categorize Iris flowers by species.
# use tensorflow's eager execution
# 1. Build a model
# 2. train this model on example data
# 3. use the model to make predictions about unknown data

# 1. Import and parse the datasets.
# 2. select the type of model.
# 3. train the model.
# 4. evaluate the model's effectiveness.
# 5. use the trained model to make predictions.

# configure imports and eager execution
# once eager execution is enabled, it cannot be disabled within the same program.
from __future__ import division, print_function, absolute_import
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print('Tensorflow version: {}'.format(tf.VERSION))
print('Eager execution: {}'.format(tf.executing_eagerly()))

# import and parse the training dataset
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url
)
print('Local copy of the dataset file: {}'.format(train_dataset_fp))

# inspect the data
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print('Features: {}'.format(feature_names))
print('Label: {}'.format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Create a tf.data.Dataset
batch_size = 32
train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)

features, labels = next(iter(train_dataset))
print(features.values())
print(labels)


def pack_features_vector(features, labels):
    '''pack the features into a single array.'''
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
print(features[:5])

# create a model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

# using the model
predictions = model(features)
print(predictions[:5])

print(tf.nn.softmax(predictions[:5]))
print('Prediction: {}'.format(tf.argmax(predictions[:5], axis=1)))
print('Labels: {}'.format(labels[:5]))


# Train the model
# define the loss and gradient function
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print('loss test: {}'.format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as t:
        loss_value = loss(model, inputs, targets)
    return loss_value, t.gradient(loss_value, model.trainable_variables)


# create an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step()

loss_value, grads = grad(model, features, labels)

print('Step: {}, initial loss: {}'.format(global_step.numpy(), loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.variables), global_step)

print('Step: {}, Loss: {}'.format(global_step.numpy(), loss(model, features, labels).numpy()))

# training loss
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # training loop
    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables), global_step)

        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print('Epoch {}: loss={}, accuracy={}.'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

# Visualize the loss function over time
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training metrics')
axes[0].set_ylabel('loss', fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# setup the test dataset
test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)

test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False
)

test_dataset = test_dataset.map(pack_features_vector)

# evaluate the model on the test dataset
test_accuracy = tfe.metrics.Accuracy()

for x, y in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print('test set accuracy: {}'.format(test_accuracy.result()))

# use the trained model to make predictions
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print('Example {} prediction: {} {}.'.format(i, name, 100*p))