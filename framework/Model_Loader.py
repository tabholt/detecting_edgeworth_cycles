'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

This file contains:
    - The Model_Container class
        - designed to load raw data sets
        - split data sets automatically into test and train data sets
        - have a simple interface 

'''

from framework import Label_Class
from framework.Estimation_Framework import Model


class Model_Container(object):
    '''
    Container class for managing training and testing datasets and models.

    This class facilitates the loading, management, and splitting of labeled
    datasets into training and testing sets for machine learning models. It
    also provides methods for handling ambiguous observations based on truth
    and false criteria.

    Args:
        region (str): The region associated with the dataset.
        truth_criterion (str): The truth criterion for labeling data.
        train_frac (float): Fraction of the data used for training.
        false_criterion (str): Optional criterion for false class labeling.
        test_on_full_set (bool): Whether to use the full dataset for testing.
        fix_seed (bool): Whether to fix random seed for reproducibility.

    Properties:
        train: Returns the training dataset.
        test: Returns the testing dataset, which may be the same as the
              training dataset for train_frac = 1.

    Methods:
        load_model: Loads the dataset and creates training and testing models.
        eliminate_ambiguous_obs: Filters out ambiguous observations based on
                                  truth and false criteria.

    '''
    def __init__(self, region, truth_criterion, train_frac, false_criterion=None, test_on_full_set=True, fix_seed=True):
        self.train_model = None
        self.test_model = None
        self.region = region
        self.train_frac = train_frac
        self.load_model(region, truth_criterion, train_frac, false_criterion,
                        test_on_full_set, fix_seed)

    @property
    def train(self):
        return self.train_model

    @property
    def test(self):
        if self.train_frac == 1:
            return self.train_model
        else:
            return self.test_model

    def load_model(self, region, truth_criterion, train_frac, false_criterion, test_on_full_set, fix_seed):
        db_locations = {
            'wa': 'label_databases/wa_label_db.json',
            'nsw': 'label_databases/nsw_label_db.json',
            'de': 'label_databases/german_label_db.json'
        }
        if region not in ['wa', 'nsw', 'de']:
            raise Exception('invalid region')
        db = Label_Class.load_label_set(db_locations[region])
        if fix_seed:
            seed = 42
        else:
            seed = None
        train, test = db.get_train_test_sets(train_frac=train_frac, seed=seed)
        if false_criterion != None:
            train = self.eliminate_ambiguous_obs(
                train, truth_criterion, false_criterion)
            test = self.eliminate_ambiguous_obs(
                test, truth_criterion, false_criterion)
        train_model = Model(train, truth_criterion, region)
        if len(test) == 0:
            test_model = Model(train, truth_criterion, region)
        elif test_on_full_set:
            test_model = Model(test + train, truth_criterion, region)
        else:
            test_model = Model(test, truth_criterion, region)

        self.train_model = train_model
        self.test_model = test_model

    def eliminate_ambiguous_obs(self, lbl_set, true_rule, false_rule):
        '''
        rules = {
            'two_yes'
            'three_yes'
            'rounded'
            'high_thresh'
            'majority_no'
            'at_least_one_no'
            'unanimous_no'
            'no_yeses'
        }
        '''
        new_lbl_set = []
        for lbl in lbl_set:
            pass_true_rule = lbl.cycling_binary_interface(true_rule)
            pass_false_rule = lbl.cycling_binary_interface(false_rule)
            if pass_true_rule or pass_false_rule:
                new_lbl_set.append(lbl)
        return new_lbl_set
