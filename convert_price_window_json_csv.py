'''
Author : Timothy Holt - tabholt@gmail.com
Sep 2023

Convert Windows JSON Data to CSV and Vice Versa.

This script provides functionality to convert data between Windows JSON and CSV
formats. It reads the input file, performs the conversion, and saves the output
file. It can handle both JSON-to-CSV and CSV-to-JSON conversions.

Usage:
    python script.py input_filename

The script accepts the following command-line arguments:
    - input_filename (str): The name of the input file (including path).
      It must be either a JSON or CSV file.

Output:
    - For JSON input, it converts the data to a CSV file with the same name.
    - For CSV input, it converts the data to a JSON file with the same name.

The script follows these steps:
    1. Load the command-line argument as the input file.
    2. Determine the file type (JSON or CSV) based on the file extension.
    3. Perform the appropriate conversion.
    4. Save the converted data to a new file, ensuring it does not overwrite an
       existing file unless the user confirms.

Note:
    - For JSON input, it assumes that the JSON file contains data in the format
      expected for conversion (with specific keys for columns).
    - For CSV input, it expects columns with specific labels (e.g., 'uuid',
      'start_date', 'post_code', 'latitude', 'longitude', 'brand', and columns
      labeled as 'p0' to 'p89' for price data).

To convert all postal regions at once use (bash):
for i in {0..9}; do python convert_price_window_json_csv.py "databases/${i}_price_windows.json"; done
'''

import pandas as pd
import json
import sys
import os

def load_argv():
    try:
        filename = sys.argv[1]
        assert os.path.exists(filename)
        print(f'Converting: {filename}')
    except:
        print(
            'Please provide file as command line argument\n    Usage: argv = [filename (including path)]\n')
        exit()
    return filename

def check_for_overwrite(export_name):
    while os.path.exists(export_name):
        textin = input(f'file: {export_name} already exists. Overwrite? (y/n): ')
        if textin.lower() == 'y':
            break
        else:
            export_name = input('Please provide new export filename (or exit): ')
            if export_name.lower() == 'exit':
                exit()
    return export_name

def load_windows_json(filename):
    with open(filename, 'r') as f:
        d = json.load(f)
    return d

def windows_dict_to_df(d):
    # columns = ['uuid', 'start_date', 'post_code', 'latitude', 'longitude', 'brand', p0, ..., p89]
    price_labels = [f'p{i}' for i in range(90)]
    det_price_labels = [f'det_p{i}' for i in range(90)]
    new_d = {}
    for i, (k, v) in enumerate(d.items()):
        subdict = {        
            'uuid' : k.split('_')[0],
            'post_code' : v['post_code'],
            'brand' : v['brand'],
            'latitude' : v['geotag'][0],
            'longitude' : v['geotag'][1],
            'start_date' : v['start_date'],
        }
        prices = dict(zip(price_labels, v['price']))
        subdict.update(prices)
        if 'detrended_price' in v:
            detrended_prices = dict(zip(det_price_labels, v['detrended_price']))
            subdict.update(detrended_prices)
        new_d[i] = subdict
    df = pd.DataFrame.from_dict(new_d, orient='index')
    df.sort_values(['post_code', 'uuid', 'start_date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


filename = load_argv()
path_name = filename.split('.')[0]
extension = filename.split('.')[-1]
if extension not in {'csv', 'json'}:
    print('Not a valid file type. File must be .csv or .json')
    exit()
if extension == 'json':
    d = load_windows_json(filename)
    df = windows_dict_to_df(d)
    print(df)
    export_name = path_name + '.csv'
    export_name = check_for_overwrite(export_name)
    df.to_csv(export_name, index=False)
elif extension == 'csv':
    df = pd.read_csv(filename, dtype={'start_date': 'str', 'post_code': 'str'})
    data_column_labels = ['uuid', 'start_date', 'post_code', 'latitude', 'longitude', 'brand']
    price_column_labels = [f'p{i}' for i in range(90)]
    det_price_column_labels = [f'det_p{i}' for i in range(90)]
    price_matrix = df[price_column_labels].to_numpy()
    if det_price_column_labels[0] in df.columns:
        det_price_matrix = df[det_price_column_labels].to_numpy()
    sub_df = df[data_column_labels]
    outer_dict = {}
    for i, obs in sub_df.iterrows():
        key = f"{obs['uuid']}_{obs['start_date']}_{len(price_column_labels)}"
        inner_dict = {
            'uuid': obs['uuid'],
            'start_date': obs['start_date'],
            'post_code': obs['post_code'],
            'geotag': (obs['latitude'], obs['longitude']),
            'brand': obs['brand'],
        }
        inner_dict['price'] = price_matrix[i,:].tolist()
        if det_price_column_labels[0] in df.columns:
            inner_dict['detrended_price'] = det_price_matrix[i,:].tolist()
        outer_dict[key] = inner_dict
    export_name = path_name + '.json'
    export_name = check_for_overwrite(export_name)
    with open(export_name, 'w') as f:
        export_json = json.dumps(outer_dict, indent=4)
        f.write(export_json)
        



