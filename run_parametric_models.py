'''
Author: Timothy Holt - tabholt@gmail.com
May 2022

Model runner script for parametric models.
To Use:
    - call this script using python
    - provide command line arguments [region, method]

    
Command Line Arguments:
    region : in {'de', 'nsw', 'wa'} - What region you would like to run an LSTM
        model for

    method : in {'PRNR', 'NMC', 'MIMD', 'MBPI', 'WAVY', 'FT0', 'FT1', 
        'FT2', 'LS0', 'LS1', 'LS2', 'CS0', 'CS1', 'all'} - if 'all' is passed
        the the a model will be build, trained, and tested for each method.



Output:
    - Will print a prediction truth table to terminal once training complete.
    - Save accuracy results to the named logfile.
'''

import os
import sys
import time
from framework.Model_Loader import Model_Container
from framework.model_settings import MAGS_default_domains
from framework.model_settings import parametric_methods
from framework.model_settings import truth_criteria as tc_dict
from framework.model_settings import full_region_names


#################### SET PARAMETERS ####################
fix_seed = True
train_fraction = .8
false_criterion = None
parametric_logger_name = 'parametric_model_log.csv'
log_sep = ','
########################################################
'''
provide argv = [region, method]

method in {
    'PRNR',
    'NMC',
    'MIMD',
    'MBPI',
    'WAVY',
    'FT0',
    'FT1',
    'FT2',
    'LS0',
    'LS1',
    'LS2',
    'CS0',
    'CS1',
    'all'
}
'''

# leave these
precision = 2
resolution = 200


def optimize_and_test_param_model(region, train_fraction, false_criterion, method, domain):
    truth_criterion = tc_dict[region]
    model = Model_Container(region, truth_criterion, train_fraction, false_criterion=false_criterion,
                            test_on_full_set=False, fix_seed=fix_seed)
    print(f'\nOptimizing {method} parameter...\n')
    theta_star, _, _ = model.train.MAGS(
        method,
        min_theta=domain[0],
        max_theta=domain[1],
        precision=precision,
        resolution=resolution,
        window_cut_factor=4,
        objective='acc',
        verbose=1
    )
    t_dict = model.test.evaluate_and_print_method(method, theta_star, domain)
    if method == 'MBPI':
        theta2 = model.train.retrieve_MBPI_theta2(theta_star)
        theta_star = [theta_star, theta2]
    tr_hash = model.train.obs_set_hash
    n_train = model.train.n
    n_test = model.test.n
    return t_dict, theta_star, n_train, n_test, tr_hash


def export_accuracy(region, method, truth_criterion, tr_hash, train_frac, truth_dict, n_train, n_test, run_time, theta_star, filename):
    perf = truth_dict['TT'] + truth_dict['FF']
    perf = perf / (perf + truth_dict['TF'] + truth_dict['FT']) * 100
    perf = round(perf, 3)
    items = {
        'region': region,
        'method': method,
        'truth_criterion': truth_criterion,
        'train_frac': train_frac,
        'n_train': n_train,
        'n_test': n_test,
        'training_set_hash': tr_hash,
        'TT': truth_dict['TT'],
        'FF': truth_dict['FF'],
        'FT': truth_dict['FT'],
        'TF': truth_dict['TF'],
        'theta_star': theta_star,
        'run_time': run_time,
        'accuracy': perf,
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


def run_method(method):
    print(f'Running {method} parametric model on {full_region_names[region]} data\n')

    domain = MAGS_default_domains[region][method]

    timer = -1 * time.perf_counter()
    t_dict, theta_star, n_train, n_test, tr_hash = optimize_and_test_param_model(
        region, train_fraction, false_criterion, method, domain)
    timer += time.perf_counter()
    timer = round(timer,1)

    export_accuracy(region, method, tc_dict[region], tr_hash, train_fraction, t_dict, n_train,
                    n_test, timer, theta_star, parametric_logger_name)



if len(sys.argv) < 3:
    raise Exception(
        'Provide argv = [region, method]')
region = sys.argv[1].lower()
method = sys.argv[2].upper()
if region not in {'de', 'nsw', 'wa'}:
    raise Exception('region must be in {de, nsw, wa}')
if method not in parametric_methods:
    raise Exception(f'method must be in {parametric_methods}')

if method == 'ALL':
    for m in parametric_methods:
        if m == 'ALL':
            continue
        run_method(m)
else:
    run_method(method)