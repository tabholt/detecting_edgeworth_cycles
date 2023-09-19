'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

Model runner script for Random Forest models.
To Use:
    - call this script using python
    - provide command line arguments [region]

    
Command Line Arguments:
    region : in {'de', 'nsw', 'wa'} - What region you would like to run an RF model for

    
Note:
    - Trained model can be saved, after training by setting parameter:
      save_model = True. In this case you should note the training set
      hash (tsh) of the saved model for future reference. 

         
Output:
    - Will print a prediction truth table to terminal once training complete.
    - Save accuracy results to the named logfile.
'''

import os
import sys
import time
from framework import RF_Framework as RF
from framework.model_settings import truth_criteria as tc_dict
from framework.model_settings import full_region_names
from framework.model_settings import rf_features


#################### SET PARAMETERS ####################
fix_random_seed = True
train_fraction = .8
false_criterion = None
rf_logger_name = 'random_forest_model_log.csv'
log_sep = ',' # log value separators
save_model = True
########################################################
# argv = [region] in {de, nsw, wa}


def build_and_train_RF(region, train_fraction, false_criterion):
    truth_criterion = tc_dict[region]
    rf = RF.RF_Model(region, truth_criterion, rf_features,
                     train_fraction, false_criterion=false_criterion, test_on_full_set=False, fix_seed=fix_random_seed)
    tr_hash = rf.training_set_hash
    rf.extract_features()
    rf.fit_model()
    if save_model:
        rf.save_model()
    t_dict = rf.test_model()
    n_train = rf.data.train.n
    n_test = rf.data.test.n
    return t_dict, n_train, n_test, tr_hash


def export_accuracy(region, truth_criterion, tr_hash, train_frac, truth_dict, n_train, n_test, run_time, filename):
    '''
       | Ground Truth | Prediction
    ===|==============|============
    TT |     True     |    True
    FF |     False    |    False
    FT |     False    |    True     * False positive
    TF |     True     |    False    * False negative
    '''
    perf = truth_dict['TT'] + truth_dict['FF']
    perf = perf / (perf + truth_dict['TF'] + truth_dict['FT']) * 100
    perf = round(perf, 3)
    items = {
        'region': region,
        'method': 'RF',
        'truth_criterion': truth_criterion,
        'train_frac': train_frac,
        'n_train': n_train,
        'n_test': n_test,
        'training_set_hash': tr_hash,
        'TT': truth_dict['TT'],
        'FF': truth_dict['FF'],
        'FT': truth_dict['FT'],
        'TF': truth_dict['TF'],
        'run_time': run_time,
        'accuracy': perf
    }
    if not os.path.isfile(filename):
        header = ''
        for d in items:
            header += d
            header += log_sep
        header += '\n'
        with open(filename, 'w') as f:
            f.write(header)

    line = ''
    for d in items:
        line += str(items[d])
        line += log_sep
    line += '\n'
    with open(filename, 'a') as f:
        f.write(line)


if len(sys.argv) < 2:
    raise Exception('provide argv = [region]')
region = sys.argv[1].lower()
if region not in {'de', 'nsw', 'wa'}:
    raise Exception('region must be in {de, nsw, wa}')


print(f'Running Random Forest model on {full_region_names[region]} data\n')

timer = -1 * time.perf_counter()
t_dict, n_train, n_test, tr_hash = build_and_train_RF(
    region, train_fraction, false_criterion)
timer += time.perf_counter()
timer = round(timer,1)


export_accuracy(region, tc_dict[region], tr_hash, train_fraction, t_dict, n_train,
                n_test, timer, rf_logger_name)
