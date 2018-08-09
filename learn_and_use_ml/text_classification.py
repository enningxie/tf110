# text classification with movie reviews
# binary classification
# imdb dataset
import tensorflow as tf
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt

# Download the IMDB dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# explore the data
print('training entries: {}, labels: {}'.format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# convert the integers back to words
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()


# the first indices are reserved
word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
word_index_reverse = {index: word for word, index in word_index.items()}


def decode_review(text):
    return ' '.join([word_index_reverse.get(index, '?') for index in text])


text_decoded = decode_review(train_data[0])
print(text_decoded)

# prepare the data
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=256
)

print(len(train_data[0]))
print(train_data[0])

# build the model
# input shape is the vocabulary count used for the movie review (10000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# loss function and optimizer
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=keras.losses.binary_crossentropy,
    metrics=['acc']
)

# create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# trian the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# evaluate the model
results = model.evaluate(test_data, test_labels)
print('test_loss: {}, test_acc: {}.'.format(results[0], results[1]))

# create a graph of acc and loss over time
# `model.fit()` returns a history object that contains a dict with everything that happend during training.
history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
loss = history_dict['loss']
val_acc = history_dict['val_acc']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)

# 'bo' is for 'blue dot'
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for 'solid blue line'
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()  # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Training and validation acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()