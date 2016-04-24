import pandas as pd
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils

from default_optimizers import def_sgd, def_adagrad, def_adadelta, def_adamax


MAIN_DIR = '/mnt1/github/claims_loss_prediction'
DATA_DIR = os.path.join(MAIN_DIR, 'data')
SUBMISSION_DIR = os.path.join(MAIN_DIR, 'submissions')


class NeuralNetModel(object):
    def __init__(self, sgd, adagrad, adadelta, adamax):
        self._model = None

        if sgd:
            self._sgd = sgd
        else:
            self._sgd = def_sgd

        if adagrad:
            self._adagrad = adagrad
        else:
            self._adagrad = def_adagrad

        if adadelta:
            self._adadelta = adadelta
        else:
            self._adadelta = def_adadelta

        if adamax:
            self._adamax = adamax
        else:
            self._adamax = def_adamax

    def binarize_data(self):
        fn = os.path.join(DATA_DIR, 'all_data_nn.csv')
        df = pd.read_csv(fn)
        df['norm_age'] = (df['age_at_ins'] - min(df['age_at_ins']))/(max(df['age_at_ins']) - min(df['age_at_ins']))
        print df.columns

        # categorical features
        cat_features = ['make', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7',
                        'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'nvcat']

        # binarize categorical features
        binarized_df1 = pd.get_dummies(df[cat_features])
        binarized_df2 = pd.get_dummies(df['ordcat'])
        binarized_df2.columns = ['ordcat_1', 'ordcat_2', 'ordcat_3', 'ordcat_4', 'ordcat_5', 'ordcat_6', 'ordcat_7']
        binarized_df = pd.concat([df[['rowid', 'var4', 'var5', 'var7', 'nvvar1', 'nvvar2', 'nvvar3', 'nvvar4',
                                      'response', 'ind', 'norm_age']],
                                  binarized_df1,
                                  binarized_df2],
                                 axis=1)

        # remove columns such that there n-1 features for a caterical variable with n values
        rem_list = ['make_Z', 'cat1_G', 'cat2_C', 'cat3_F', 'cat4_C', 'cat5_C', 'cat6_F', 'cat7_D',
                    'cat8_C', 'cat9_B', 'cat10_C', 'cat11_F', 'cat12_F', 'nvcat_O', 'ordcat_7']
        binarized_df = binarized_df.drop(rem_list, axis=1)
        binarized_df.to_csv('all_data_nn_binarized.csv', index=False, index_label=False)

    def load_train_test(self):
        binarized_df = pd.read_csv(os.path.join(DATA_DIR, 'all_data_nn_binarized.csv'))
        train = binarized_df[binarized_df.ind == 'train']
        test = binarized_df[binarized_df.ind == 'test']

        cols = [col for col in train.columns if col not in ['rowid', 'response', 'ind']]

        self.X = train[cols].values.copy()
        self.y = np_utils.to_categorical(train['response'], 2)
        self.X_test = test[cols].values.copy()
        self.X_test_rowids = test['rowid'].values.copy()

        self.X = self.X.astype(np.float32)
        self.y = self.y, self.X_test.astype(np.float32)

    def build_network(self):
        self._model = Sequential()

        # Add 1st hidden layer; it should also accept input vector dimension
        # Dense is a fully connected layer with tanh as the activation function
        self._model.add(Dense(256, input_shape=(self.X.shape[1]), init='uniform', activation='tanh'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(256, activation='tanh'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(256, activation='tanh'))
        self._model.add(Dropout(0.5))

        # output layer with sigmoid activation
        self._model.add(Dense(2, activation='sigmoid'))

    def train_and_predict(self, opt, fn):
        self._model.compile(loss='binary_crossentropy', optimizer=opt)
        self._model.fit(self.X, self.y, nb_epoch=200, validation_split=0.20, show_accuracy=True)
        proba = self._model.predict_proba(self.X_test, batch_size=32)
        # pred_cls = model.predict_classes(X_test, batch_size=32)
        result = pd.concat([
                    pd.DataFrame(self.X_test_rowids, columns=['RowId']),
                    pd.DataFrame(proba[:, 1], columns=['ProbabilityOfResponse'])], axis=1)
        result.to_csv(fn, index=False)

    def execute(self, binarize=False):
        if binarize:
            print '----- binarize data -----'
            self.binarize_data()

        print '----- create train/test data -----'
        self.load_train_test()

        print('----- Building model -----')
        self.build_network()

        print('----- Training SGD model -----')
        self.train_and_predict(self._sgd, os.path.join(SUBMISSION_DIR, 'nn_sgd_2.csv'))

        print('----- Training Adagrad model -----')
        self.train_and_predict(self._adagrad, os.path.join(SUBMISSION_DIR, 'nn_adagrad_2.csv'))

        print('----- Training Adadelta model -----')
        self.train_and_predict(self._adadelta, os.path.join(SUBMISSION_DIR, 'nn_adadelta_2.csv'))

        print('----- Training Adamax model -----')
        self.train_and_predict(self._adamax, os.path.join(SUBMISSION_DIR, 'nn_adamax_2.csv'))


def main():
    nn = NeuralNetModel()
    nn.execute()

if __name__ == '__main__':
    main()
