'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

This file contains:
    - The definition of the RF network architecture
    - The wrappers for building the necessary data arrays to 
      train and test model
    - The wrappers to build, train, and test the RF models

Notes:
    - Init-variable 'feature_list' should be is populated with a 
      set of features to use for RF model.
    - Option to save and load trained models - this is somewhat 
      robust to versions of SKLearn library used.

'''

import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from framework.Model_Loader import Model_Container

np.set_printoptions(suppress=True)  # supress scientific notation


class RF_Model(object):
    def __init__(self, region, truth_criterion, feature_list, train_frac, false_criterion=None, test_on_full_set=True, fix_seed=True):
        self.data = Model_Container(
            region, truth_criterion, train_frac, false_criterion, test_on_full_set, fix_seed)
        self.region = region
        self.train_frac = train_frac
        self.model = None
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.feature_list = feature_list
        self.saved_network_dir = 'pretrained_models/rf_models/'

    @property
    def training_set_hash(self):
        return self.data.train.obs_set_hash

    @property
    def testing_set_hash(self):
        return self.data.test.obs_set_hash

    @property
    def accuracy(self):
        n_correct = np.count_nonzero(self.predictions == self.y_te)
        pct_correct = (n_correct/len(self.y_te))*100
        return pct_correct

    @property
    def predictions(self):
        return self.model.predict(self.X_te)
    
    @property
    def network(self):
        return self.model

    def make_test_set_100_pct(self):
        print('Converting test set to 100 percent sample...')
        self.extract_features()
        print(self.X_te.shape)
        print(self.X_tr.shape)
        self.X_te = np.concatenate((self.X_tr, self.X_te), axis=0)
        print(self.X_te.shape)
        print(self.y_te.shape)
        print(self.y_tr.shape)
        self.y_te = np.concatenate((self.y_tr, self.y_te), axis=0)
        print(self.y_te.shape)

    def extract_features(self):
        print('extracting features...')
        self.X_tr, self.y_tr = self.get_features('train')
        self.X_te, self.y_te = self.get_features('test')

    def get_features(self, train_or_test):
        if train_or_test == 'train':
            data = self.data.train
        else:
            data = self.data.test
        X = np.ndarray((data.n, len(self.feature_list)))
        for i, feat in enumerate(self.feature_list):
            if feat == 'MBPI':
                X[:, i] = data.g_MBPI(data.MBPI_theta1_star)
            else:
                data.calc_method_array(feat)
                X[:, i] = data.method_arrays[feat]
        y = data.truth
        return X, y

    def get_truth(self, train_or_test):
        if train_or_test == 'train':
            return self.y_tr
        else:
            return self.y_te

    def fit_model(self):
        model = RandomForestClassifier()
        # model = AdaBoostClassifier()
        print('fitting model...')
        model.fit(self.X_tr, self.y_tr)
        self.model = model

    def test_model(self):
        print('testing model...')
        y_truth = self.y_te
        y_pred = self.model.predict(self.X_te)
        truth_dict = self.evaluate_truth(y_truth, y_pred)
        heading = f'{self.region.upper()} - Random Forest - train_frac = {self.train_frac}'
        self.print_truth_table(truth_dict, heading)
        return truth_dict

    def save_model(self):
        path = f'{self.saved_network_dir}{self.region}/{self.training_set_hash}/'
        print(f'saving rf model to: {path}\n')
        os.makedirs(path)
        with open(path + 'RF_Model.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

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
