'''
Author: Timothy Holt - tabholt@gmail.com
Sep 2023

External Classification Using Parametric Model Optimal Thetas.

This script loads external price data, evaluates parametric models using
optimal theta values, and exports classification results to JSON or CSV files.

To Use:
    - change external_data_path to chosen external data file in PARAMETERS section
    - call this script using python
    - provide command line arguments [region, method]

Command Line Arguments:
    region : in {'de', 'nsw', 'wa'} - What region you would like to run an LSTM
        model for

    method : in {'PRNR', 'MIMD', 'NMC', 'MBPI', 'FT0', 'FT1', 
        'FT2', 'LS0', 'LS1', 'LS2', 'CS0', 'CS1', 'WAVY', 'all'} - if 'all' is passed
        the the a model will be build, trained, and tested for each method.

External Data Formats:
- JSON Input: 'external_data_path' a JSON file containing price data.
- CSV Input: 'external_data_path' a CSV file with price data columns.

Results are saved as 'region_method_results.json' or 'region_method_results.csv',
depending on the chosen file type.

Make sure to run 'run_parametric_model.py' to generate optimal theta values
before using this script.

Note:
    - This script assumes that optimal thetas exist for the specified
      region and model type.
    - The external data must be properly formatted according to the chosen
      input format (JSON or CSV).
    - The classification results are saved in either JSON or CSV format
      based on the specified filename.
'''

import os
import sys
import json
import numpy as np
import pandas as pd
from framework.Model_Loader import Model_Container
from framework.model_settings import parametric_methods

np.set_printoptions(edgeitems=8, precision=4, suppress=True, linewidth=180)

#################### SET PARAMETERS ####################
# basic settings (should be updated)
# external_data_path = 'label_databases/ALL_detrended_price_windows.json' # either json or csv
external_data_path = 'label_databases/nsw_label_db.json' # either json or csv

# advanced settings (recommend to not update)
results_export_suffix = 'external_classification_results.json' # either json or csv
optimal_thetas_filename = 'pretrained_models/parametric_models/optimal_thetas.json'
series_prefix = 'det_p' # for csv input: prefix to numbered data columns
id_column_name = 'uuid' # for csv input: column name for unique identifiers
series_name = 'detrended_price' # for json input: name of data field
########################################################
'''
provide argv = [region, method]

method in {
    'PRNR',
    'MIMD',
    'NMC',
    'MBPI',
    'FT0',
    'FT1',
    'FT2',
    'LS0',
    'LS1',
    'LS2',
    'CS0',
    'CS1',
    'WAVY',
    'all'
}
'''

def load_theta_star(region, method):
    if not os.path.exists(optimal_thetas_filename):
        no_theta_star_file = f'No optimal thetas file found at {optimal_thetas_filename}. Use run_parametric_model.py to save optimal theta values.'
        raise Exception(no_theta_star_file)
    with open(optimal_thetas_filename, 'r') as f:
        params = json.load(f)
    no_theta_star = f'{region} {method} optimal theta values not found in {optimal_thetas_filename}. Use run_parametric_model.py to save optimal theta values.'
    if region not in params:
        raise Exception(no_theta_star)
    if method not in params[region]:
        raise Exception(no_theta_star)
    return params[region][method]


def convert_json_price_matrix(external_data_path, series_name='detrended_price'):
    with open(external_data_path, 'rb') as json_file:
        db = json.load(json_file)
    price_matrix = np.ndarray((len(db), 90))
    for i, ob in enumerate(db.values()):
        price_matrix[i, :] = ob[series_name]
    col_labels = list(db.keys())
    return col_labels, price_matrix

def convert_csv_price_matrix(external_data_path, series_prefix='det_p', id_column_name='uuid'):
    df = pd.read_csv(external_data_path)
    series_column_labels = [f'{series_prefix}{i}' for i in range(90)]
    price_matrix = df[series_column_labels].to_numpy()
    col_labels = df[id_column_name].tolist()
    if id_column_name == 'uuid' and 'start_date' in df.columns:
        dates = df['start_date'].to_list()
        col_labels = [f'{uuid}_{dates[i]}_90' for i, uuid in enumerate(col_labels)]
    return col_labels, price_matrix


def evaluate_method_external_data(method, col_labels, price_matrix, theta_star):
    m = Model_Container('nsw', 'three_yes', .01,
                            false_criterion=None,
                            test_on_full_set=False, fix_seed=True).train
    m.evaluate_external_data(price_matrix, method)
    eval_func = m.method_eval_dict[method]
    evals = eval_func(theta_star)
    return dict(zip(col_labels, evals))
    


def export_results(dictionary, filename, region, method):
    filename = f'{region}_{method}_{filename}'
    print(f'Saving classification results to: {filename}')
    if filename.split('.')[-1] == 'json':
        dictionary = {k: int(v) for k, v in dictionary.items()}
        with open(filename, 'w') as json_f:
            export_json = json.dumps(dictionary, indent=4)
            json_f.write(export_json)
    elif filename.split('.')[-1] == 'csv':
        df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['classification'])
        df.to_csv(filename, index_label=id_column_name)
    else:
        raise Exception('Incompatible export file type. Must be json or csv.')


def run_method(region, method, col_labels, price_matrix):
    theta_star = load_theta_star(region, method)
    print(f'\nEvaluating {method} on external data with theta={theta_star}')
    classifications = evaluate_method_external_data(method, col_labels, price_matrix, theta_star)
    export_results(classifications, results_export_suffix, region, method)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception(
            'Provide argv = [region, method]')
    region = sys.argv[1].lower()
    method = sys.argv[2].upper()
    if region not in {'de', 'nsw', 'wa'}:
        raise Exception('region must be in {de, nsw, wa}')
    if method not in parametric_methods:
        raise Exception(f'method must be in {parametric_methods}')

    print(f'loading data from: {external_data_path}...')
    if external_data_path.split('.')[-1] == 'json':
        col_labels, price_matrix = convert_json_price_matrix(
            external_data_path,
            series_name=series_name,
            )
    elif external_data_path.split('.')[-1] == 'csv':
        col_labels, price_matrix = convert_csv_price_matrix(
            external_data_path,
            series_prefix=series_prefix,
            id_column_name=id_column_name
            )
    else:
        raise Exception('Incompatible file type. Must be json or csv.')

    if method == 'ALL':
        for m in parametric_methods:
            if m == 'ALL':
                continue
            run_method(region, m, col_labels, price_matrix)
    else:
        run_method(region, method, col_labels, price_matrix)