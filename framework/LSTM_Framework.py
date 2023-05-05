'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

This file contains:
    - The definition of the LSTM network architecture
    - The wrappers for building the necessary data arrays to 
      train and test model
    - The wrappers to build, train, and test the LSTM models

Notes:
    - Init-variable 'feature_list' will define whether it is ensemble or basic
      model. If feature_list is empty then it will be basic, if feature_list
      is populated with a set of features it will automatically build based
      on those features.
    - In the paper 100 epochs was used for all training
    - Network architecture is defined in LSTM_Model.build_network
    - Option to save and load trained models - this is sensetive to versions of
      TensorFlow library used.

'''

import numpy as np
from collections import defaultdict
import tensorflow as tf
from framework.Model_Loader import Model_Container
import time
import random
import os
import pickle


class LSTM_Model(object):
    def __init__(self, region, truth_criterion, feature_list, train_frac, false_criterion=None, normalize=False, test_on_full_set=True, fix_seed=True):
        self.data = Model_Container(
            region, truth_criterion, train_frac, false_criterion, test_on_full_set, fix_seed)
        self.region = region
        self.train_frac = train_frac
        self.feature_list = feature_list
        self.network = None
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.normalize = normalize
        self.saved_network_dir = 'pretrained_models/lstm_models/'

    @property
    def training_set_hash(self):
        return self.data.train.obs_set_hash

    @property
    def testing_set_hash(self):
        return self.data.test.obs_set_hash

    @property
    def accuracy(self):
        evals_te = self.predictions
        n_correct = np.count_nonzero(evals_te == self.y_te)
        pct_correct = (n_correct/len(self.y_te))*100
        return pct_correct

    @property
    def predictions(self):
        evals_te = self.network.predict(self.X_te)
        evals_te = tf.where(evals_te <= 0.5, 0, 1)
        return evals_te

    def build_network(self):
        dropout_val = 0.5
        act_fun = 'tanh'
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=.0005)  # default=.001
        model = tf.keras.models.Sequential(name=self.region)
        model.add(tf.keras.layers.LSTM(
            units=16, activation=act_fun, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_val))
        model.add(tf.keras.layers.LSTM(
            units=8, activation=act_fun, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_val))
        model.add(tf.keras.layers.LSTM(units=4, activation=act_fun,))
        model.add(tf.keras.layers.Dropout(dropout_val))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,  # 'rmsprop'
                      metrics=['accuracy'])
        self.network = model

    def extract_features(self):
        timer = -1 * time.perf_counter()
        print('extracting training features...')
        self.X_tr = self.build_X('train')
        self.y_tr = self.data.train.truth
        print('extracting testing features...')
        self.X_te = self.build_X('test')
        self.y_te = self.data.test.truth
        timer += time.perf_counter()
        print(f'features extracted in {timer:.1f} s\n')
        print(f'X_tr shape = {self.X_tr.shape}')
        print(f'X_te shape = {self.X_te.shape}')

    def make_test_set_100_pct(self):
        print('Converting test set to 100 percent sample...')
        self.X_te = np.concatenate((self.X_tr, self.X_te), axis=0)
        self.y_te = np.concatenate((self.y_tr, self.y_te), axis=0)

    def build_X(self, train_or_test):
        if train_or_test == 'train':
            data = self.data.train
        elif train_or_test == 'test':
            data = self.data.test
        if len(self.feature_list) > 0:
            for i, feat in enumerate(self.feature_list):
                if feat == 'MBPI':
                    continue
                else:
                    data.calc_method_array(feat)
            features = data.method_arrays
            features['LS'] = data.LS_array
            if 'MBPI' in self.feature_list:
                features['MBPI'] = data.g_MBPI(data.MBPI_theta1_star)
        else:
            features = {}
        features['delta_p'] = data.delta_price_array
        X = np.ndarray((data.n, 90, len(features)))
        for i, feat in enumerate(features):
            f = features[feat]
            if len(f.shape) == 1:
                x = np.ndarray((data.n, 90))
                x.fill(0)
                x[:, 0] = f
            elif f.shape[1] < 90:
                a = np.ndarray((data.n, 90-f.shape[1]))
                a.fill(0)
                x = np.concatenate((f, a), axis=1)
            else:
                x = f
            X[:, :, i] = x
        if self.normalize:
            X = self.normalize_X(X)
        return X

    def normalize_X(self, X):
        for k in range(X.shape[2]):
            x = X[:, :, k]
            k_min = np.amin(x)
            k_max = np.amax(x)
            if (x[:, 1:] == 0).all():
                x = x[:, 0]
                X[:, 0, k] = (x - k_min) / (k_max - k_min)
            else:
                X[:, :, k] = (x - k_min) / (k_max - k_min)
        return X

    def load_pretrained_network(self):
        fname = self.saved_network_dir
        if self.feature_list == []:
            fname += 'basic/'
        else:
            fname += 'ensemble/'
        fname += f'{self.region}/{self.training_set_hash}/network'
        print(f'loading TF network: {fname}')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
        self.network = tf.keras.models.load_model(fname)

    def save_network(self):
        path = self.saved_network_dir
        if self.feature_list == []:
            path += f'basic/{self.region}/{self.training_set_hash}/network'
        else:
            path += f'ensemble/{self.region}/{self.training_set_hash}/network'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
        self.network.save(path)

    def save_model(self):
        print('Saving model...')
        path = self.saved_network_dir
        if self.feature_list == []:
            path += f'basic/{self.region}/{self.training_set_hash}/'
        else:
            path += f'ensemble/{self.region}/{self.training_set_hash}/'
        os.makedirs(path)
        temp = self.network
        self.network = None
        with open(path + 'LSTM_Model.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.network = temp
        self.save_network()

    def train(self, epochs, batch_size):
        print('Training LSTM...')
        run_id = str(random.random())[2:]
        checkpoint_filename = f'/tmp/checkpoint/{self.region}_train-{str(int(self.train_frac*100))}_{run_id}'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filename,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        self.network.fit(x=self.X_tr, y=self.y_tr, epochs=epochs,
                         validation_data=(self.X_te, self.y_te), batch_size=batch_size, callbacks=[cp_callback])
        self.network.load_weights(checkpoint_filename)

    def evaluate_model(self):
        print('Out of sample performance')
        self.network.evaluate(self.X_te, self.y_te)
        self.network.summary()

    def predict(self):
        evals_te = self.network.predict(self.X_te)
        evals_te = tf.where(evals_te <= 0.5, 0, 1)
        t_dict = self.evaluate_truth(
            self.y_te, evals_te)
        self.print_truth_table(t_dict, 'LSTM - ' + self.region)
        return t_dict

    @property
    def accuracy(self):
        evals_te = self.predictions
        n_correct = np.count_nonzero(evals_te == self.y_te)
        pct_correct = (n_correct/len(self.y_te))*100
        return pct_correct

    @property
    def predictions(self):
        evals_te = self.network.predict(self.X_te)
        evals_te = tf.where(evals_te <= 0.5, 0, 1)
        evals_te = np.array(evals_te, dtype='bool').flatten()
        return evals_te

    def evaluate_truth(self, truth, evaluations):
        '''
        first letter represents truth
        second letter represents evaluation
        '''
        truth_dict = defaultdict(int)
        for i in range(len(evaluations)):
            t = truth[i]
            e = evaluations[i]
            if t == True and e == True:
                truth_dict['TT'] += 1
            elif t == False and e == False:
                truth_dict['FF'] += 1
            elif t == False and e == True:
                truth_dict['FT'] += 1  # Type 1 error (false positive)
            elif t == True and e == False:
                truth_dict['TF'] += 1  # Type 2 error (false negative)
        return truth_dict

    def print_truth_table(self, truth_dict, heading=' '):
        col_widths = [17, 15, 15]
        perf = truth_dict['TT'] + truth_dict['FF']
        perf = perf / (perf + truth_dict['TF'] + truth_dict['FT']) * 100
        # subheading = f'truth criterion = {self.truth_criterion}'
        print('\n{:^{}}'.format(heading, np.sum(col_widths)))
        # print('{:^{}}'.format(subheading, np.sum(col_widths)))
        print('{:>{}}{:>{}}'.format(
            ' ', col_widths[0], 'METHOD EVALUATIONS', col_widths[1]+col_widths[2]))
        print('{:<{}}|{:>{}}{:>{}}'.format(
            'GROUND TRUTH', col_widths[0]-1, 'Cycling', col_widths[1], 'Not_Cycling', col_widths[2]))
        print('='*(np.sum(col_widths)))
        print('{:<{}}|{:>{}}{:>{}}'.format(
            'Cycling', col_widths[0]-1, truth_dict['TT'], col_widths[1], truth_dict['TF'], col_widths[2]))
        print('{:<{}}|{:>{}}{:>{}}'.format(
            'Not_Cycling', col_widths[0]-1, truth_dict['FT'], col_widths[1], truth_dict['FF'], col_widths[2]))
        print('-'*(np.sum(col_widths)))
        print(f'ACCURACY = {perf:.2f}%\n')
