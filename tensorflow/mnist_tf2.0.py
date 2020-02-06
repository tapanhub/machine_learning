import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


EPOCHS = 10
BATCH_SIZE = 128
VERBOSE = 0
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2



#load mnist dataset

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
#reshape input
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')

#normalize input

X_train /= 255
X_test /= 255

#one hot representation of the output

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

#build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES, input_shape=(784,),
                            name='dense_layer', activation='softmax'))

#compile the model
model.compile(optimizer='SGD', loss='categorical_crossentropy',
              matrics = [ 'accuracy'])

#train the model
ep_vs_acc = []
for i in range(1, EPOCHS):
    model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS,
         verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

#evaluate the model

#test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    test_loss = model.evaluate(X_test, Y_test)
    print("test loss = {}, test accuracy = {}".format(test_loss, 1-test_loss))
    ep_vs_acc.append(1-test_loss)


# Data for plotting
epoch = range(1,EPOCHS)
accuracy = ep_vs_acc

fig, ax = plt.subplots()
ax.plot(epoch, accuracy)

ax.set(xlabel='epoch', ylabel='accuracy',
       title='epoch vs accuracy')
ax.grid()

fig.savefig("epoch_vs_accuracy.png")
plt.show()
