'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

Model runner script for LSTM models.
To Use:
    - call this script using python
    - provide command line arguments [region, training_epochs, ensemble_bool]

    
Command Line Arguments:
    region : in {'de', 'nsw', 'wa'} - What region you would like to run an LSTM model for

    training_epochs : positive integer - the number of epochs for which you 
        would like to train the LSTM model. An epoch is a single run through
        the data set. Model fit will increase with the number of epochs until 
        over-fitting is achieved. 
        For the paper 100 epochs was used, less than 10 is not recommended. 

    ensemble_bool : in {0, 1} - whether to use ensemble LSTM model or Basic one
        0 will give basic model, 1 will give ensemble model

        
Note:
    - Trained model can be saved, after training by setting parameter:
      save_model = True. In this case you should note the training set
      hash (tsh) of the saved model for future reference. 
      

Output:
    - Will print TensorFlow data to the terminal as well as a prediction truth
      table once training is complete.
    - Save accuracy results to the named logfile.
'''

import os
import sys
import time
from framework import Label_Class
from framework import LSTM_Framework as LSTM
from framework.model_settings import truth_criteria as tc_dict
from framework.model_settings import full_region_names
from framework.model_settings import lstm_features

#################### SET PARAMETERS ####################
fix_seed = True
detrend_price = True
train_fraction = .8
false_criterion = None
batch_size = 30
lstm_logger_name = 'lstm_model_log.csv'
log_sep = ',' # log value separators
save_model = True
########################################################
'''
provide argv = [region, training_epochs, ensemble_bool]
'''


def build_and_train_LSTM(region, train_fraction, detrend_price, false_criterion, EPOCHS):
    truth_criterion = tc_dict[region]
    model = LSTM.LSTM_Model(region,
                            truth_criterion,
                            lstm_features,
                            train_fraction,
                            detrend_price=detrend_price,
                            false_criterion=false_criterion,
                            normalize=False,
                            test_on_full_set=False,
                            fix_seed=fix_seed
                            )
    tr_hash = model.training_set_hash
    print(tr_hash)
    model.build_network()
    model.extract_features()
    model.train(EPOCHS, batch_size)
    # weights = model.network.get_weights()
    model.evaluate_model()
    if save_model:
        model.save_model()
    t_dict = model.predict()
    n_train = model.data.train.n
    n_test = model.data.test.n
    return t_dict, n_train, n_test, tr_hash


def export_accuracy(region, truth_criterion, tr_hash, train_frac, truth_dict, n_train, n_test, epochs, run_time, filename):
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
        'method': 'LSTM',
        'truth_criterion': truth_criterion,
        'train_frac': train_frac,
        'n_train': n_train,
        'n_test': n_test,
        'epochs': epochs,
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


if len(sys.argv) < 3:
    raise Exception(
        'provide argv = [region, training_epochs, ensemble_bool]')
if sys.argv[3] not in ['1', '0']:
    raise Exception('ensemble_bool must be either 0 (False) or 1 (True)')
region = sys.argv[1].lower()
epochs = int(sys.argv[2])
ensemble = int(sys.argv[3])  # expect 1 or 0 (bool)
if region not in {'de', 'nsw', 'wa'}:
    raise Exception('region must be in {de, nsw, wa}')
ensemble_name = {1: 'Ensemble', 0: 'Basic'}
print(f'Running {ensemble_name[ensemble]} LSTM model for {epochs} training epochs on {full_region_names[region]} data\n')



if not ensemble:
    lstm_features = []
    lstm_logger_name = 'basic_' + lstm_logger_name
else:
    lstm_logger_name = 'ensemble_' + lstm_logger_name


timer = -1 * time.perf_counter()
t_dict, n_train, n_test, tr_hash = build_and_train_LSTM(
    region, 
    train_fraction, 
    detrend_price=detrend_price, 
    false_criterion=None, 
    EPOCHS=epochs
)
timer += time.perf_counter()
timer = round(timer,1)

export_accuracy(region, tc_dict[region], tr_hash, train_fraction, t_dict, n_train,
                n_test, epochs, timer, lstm_logger_name)