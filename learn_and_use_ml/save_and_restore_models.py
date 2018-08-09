# save and restore models

import os
import tensorflow as tf
from tensorflow import keras
import pathlib

# Get an example dataset
# To speed up process, only use the first 1000 examples.
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0


# Define a model
# Returns a short sequential model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['acc']
    )

    return model


model = create_model()
model.summary()

# 1
# Save checkpoints during training
# Checkpoint callback usage
checkpoint_path = './training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)

model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]  # pass callback to training
)

# restore model from checkpoint
model_new = create_model()
loss, acc = model_new.evaluate(test_images, test_labels)
print('Untrained model, accuracy: {:5.2f}%'.format(100*acc))

model_new.load_weights(checkpoint_path)
loss_new, acc_new = model_new.evaluate(test_images, test_labels)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc_new))

# 2
# checkpoint callback options
# train a new model, and save uniquely named checkpoints once every 5-epochs
checkpoint_path_new = './training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir_new = os.path.dirname(checkpoint_path_new)

cp_callback_new = keras.callbacks.ModelCheckpoint(
    checkpoint_path_new, verbose=1, save_weights_only=True,
    # save weights, every 5 epochs
    period=5
)

model_new_ = create_model()
model_new_.fit(train_images, train_labels,
               epochs=50, callbacks=[cp_callback_new],
               validation_data=(test_images, test_labels),
               verbose=1)

# sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir_new).glob('*.index')
checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
print(checkpoints)

# reset the model and load the latest checkpoint.
model_latest = create_model()
model.load_weights(latest)
loss_latest, acc_latest = model_latest.evaluate(test_images, test_labels)
print('Latest model, acc: {:5.2f}%'.format(100*acc_latest))

# 3
# manually save weights
# save the weights
model_latest.save_weights('./checkpoints/latest')

# restore the weights
model_ = create_model()
model_.load_weights('./checkpoints/latest')
loss_, acc_ = model_latest.evaluate(test_images, test_labels)
print('model_, acc: {:5.2f}%'.format(100*acc_latest))

# 4
# save the entire model
model_entire = create_model()
model_entire.fit(train_images, train_labels, epochs=5)
model_entire.save('./checkpoints/model_entire.h5')

# recreate the model from that file.
# recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('./checkpoints/model_entire.h5')
new_model.summary()
new_loss, new_acc = new_model.evaluate(test_images, test_labels)
print('new model, acc: {:5.2f}%'.format(100*new_acc))

# warning: Keras saves models by inspecting the architecture.
# Currently, it is not able to save TensorFlow optimizers (from tf.train).
# When using those you will need to re-compile the model after loading,
# and you will loose the state of the optimizer.