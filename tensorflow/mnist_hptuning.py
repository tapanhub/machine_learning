import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

#for tensorboard
import os

# network and training
EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3

#The hyperparameters & their values to be tested are stored in aspecial type
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256, 512]))

#Settinf the Metric to RMSE
#METRIC_RMSE = 'RootMeanSquaredError'
METRIC_RMSE = 'accuracy'

#Clear any logs from previous runs
#!rm -rf ./logs/


#A function that trains and validates the model and returns the rmse
def train_test_model(hparams):
#Keras sequential model with Hyperparameters passed from the argument
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Dense(hparams[HP_NUM_UNITS],
            input_shape=(RESHAPED,),
            name='dense_layer', activation='relu'))
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(keras.layers.Dense(hparams[HP_NUM_UNITS],
            name='dense_layer_2', activation='relu'))
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(keras.layers.Dense(NB_CLASSES,
            name='dense_layer_3', activation='softmax'))
# summary of the model
    model.summary()


#Compiling the model
    model.compile(
            optimizer=hparams[HP_OPTIMIZER],
            loss='categorical_crossentropy',
            metrics=['accuracy']
    )

#Training the network
    model.fit(X_train, Y_train,
        batch_size=hparams[HP_BATCHSIZE], epochs=EPOCHS,
        verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    test_loss, test_acc = model.evaluate(X_test, Y_test)
    return test_acc

#A function to log the training process
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        test_acc = train_test_model(hparams)
        tf.summary.scalar(METRIC_RMSE, test_acc, step=10)



# loading MNIST dataset
# verify
# the split between train and test is 60,000, and 10,000 respectly
# one-hot is automatically applied
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize in [0,1]
X_train, X_test = X_train / 255.0, X_test / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#one-hot
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Training the model for each combination of the hyperparameters.

#A unique number for each training session
session_num = 0

#Nested for loop training with all possible  combinathon of hyperparameters
for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in tf.linspace(HP_DROPOUT.domain.min_value,HP_DROPOUT.domain.max_value,3):
        for optimizer in HP_OPTIMIZER.domain.values:
            for batchsize in HP_BATCHSIZE.domain.values:
                hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: float("%.2f"%float(dropout_rate)), # float("%.2f"%float(dropout_rate)) limits the decimal palces to 2
                        HP_OPTIMIZER: optimizer,
                        HP_BATCHSIZE: batchsize,
                    }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1


# making prediction
#predictions = model.predict(X_test)
