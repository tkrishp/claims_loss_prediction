import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def load_claims_data():
    binarized_df = pd.read_csv('all_data_nn_binarized.csv')
    train = binarized_df[binarized_df.ind == 'train']
    test = binarized_df[binarized_df.ind == 'test']

    cols = [col for col in train.columns if col not in ['rowid', 'response', 'ind']]

#   np.random.shuffle(train)
    X_train = train[cols].values.copy()
    X_train = X_train.astype('float32')
    y_train = np_utils.to_categorical(train['response'], 2)
    X_test = test[cols].values.copy()
    X_test = X_test.astype('float32')

    return [X_train, X_test, y_train]


def load_data():
    print 'Loading data...'
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))

    print 'Data loaded.'
    return [X_train, X_test, y_train, y_test]


def init_llaims_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(128, input_dim=113))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    print 'Model compield in {0} seconds'.format(time.time() - start_time)
    return model


def init_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(500, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    print 'Model compield in {0} seconds'.format(time.time() - start_time)
    return model


def run_network(data=None, model=None, epochs=20, batch=256, type=None):
    try:
        start_time = time.time()

        if type == 'claims':
            X_train, X_test, y_train = load_claims_data()
        else:
            if data is None:
                X_train, X_test, y_train, y_test = load_data()
            else:
                X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history], show_accuracy=True, verbose=2)

        #print "Training duration : {0}".format(time.time() - start_time)
        #score = model.evaluate(X_test, y_test, batch_size=16, show_accuracy=True)

        #print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses


def main():
    run_network()


if __name__ == '__main__':
    main()