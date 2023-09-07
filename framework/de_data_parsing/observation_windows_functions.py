'''
Author: Timothy Holt - tabholt@gmail.com
Aug 2023

Functions for generating and exporting a test dataset of fuel price observations
into 90 day windows of average daily price observations.


Parameters:
    - dirs (dict): A dictionary containing directory paths for various data
      sources. It should include the following keys:
        - 'database_directory': The directory where the output dataset will
          be saved.
        - 'price_window_filename': The base filename for the output JSON
          dataset.
    - start_date (datetime.date): The start date for the data collection
      window.
    - end_date (datetime.date): The end date for the data collection window.
    - sparsity_threshold (int): The minimum number of days between
      observations to include in the dataset.
    - postal_region (int): The postal region code to filter the data.

    
Functions:
    - load_pickle_db(dirs, postal_region): Loads the pickled fuel price
      database for the specified postal region.
    - parse_db_populate_obs_dict(dirs, postal_region, major_q_range,
      obs_counter): Generates a dictionary of quarterly observations based
      on specified criteria.
    - generate_test_set_dict(db, test_set_data_keys, sparsity_threshold,
      postal_region): Generates a dictionary of test set data based on the
      filtered criteria.
    - gen_export_dict(test_set_dict): Generates a dictionary in a specific
      format for export.
    - main(dirs, start_date, end_date, sparsity_threshold, postal_region):
      The main function that orchestrates the data generation and export.

      
Note:
    - This script processes fuel price data and generates a test dataset
      with specific criteria for further analysis.
    - Ensure that the required external modules and classes are properly
      imported.
'''

import os
import gc
import json
import pickle
import datetime
import pandas as pd


def load_pickle_db(dirs, postal_region):
    database_directory = dirs['database_directory']
    price_db_filename = dirs['price_db_filename']
    db_filename = f'{str(postal_region)}_{price_db_filename}'
    db_filename = os.path.join(database_directory, db_filename)
    with open(db_filename, 'rb') as pickle_file:
        db = pickle.load(pickle_file)
    return db


def parse_db_populate_obs_dict(dirs, postal_region, major_q_range, obs_counter):
    '''
    run first to get dictionary of all quarterly observations
    the dictionary has:
        - keys: obs number (from 0 to n)
        - values: tuple (postal_region, station_uuid, quarter_start_date)
    '''
    # keys are observation number, values are tuple (postal_region, station_id, quarter_start_date)
    complete_obs_dict = {}
    db = load_pickle_db(dirs, postal_region)
    for uuid in db:
        first_ts = db[uuid].timestamp[0]
        last_ts = db[uuid].timestamp[-1]
        start_q_idx = -1
        end_q_idx = -1
        for q_idx, q in enumerate(major_q_range):
            if first_ts > q:
                continue
            elif start_q_idx == -1:
                start_q_idx = q_idx
            elif last_ts > q + datetime.timedelta(days=92):
                end_q_idx = q_idx
        if start_q_idx == -1 or end_q_idx == -1:  # dud station has less than 1Q of data
            continue
        for q_idx in range(start_q_idx, end_q_idx+1):
            q_s_date = major_q_range[q_idx].date()
            complete_obs_dict[obs_counter] = (postal_region, uuid, q_s_date)
            obs_counter += 1

    return obs_counter, complete_obs_dict, db


def generate_test_set_dict(db, test_set_data_keys, sparsity_threshold, postal_region):
    '''
    run second to generate the station dict with observations
    to put into test set.
    '''
    test_set_dict = {}
    for i, ob in enumerate(test_set_data_keys):
        prefix = ob[0]
        uuid = ob[1]
        q_start_date = ob[2]
        if prefix != postal_region:
            continue
        trimmed_report = db[uuid].trim(q_start_date, days=90)
        if trimmed_report == db[uuid]:
            continue  # occurs when station has gap in reporting
        if trimmed_report.n < 90/sparsity_threshold:
            continue
        if trimmed_report.days > 95:  # these reports have data problems
            continue
        test_set_dict[trimmed_report.uuid] = trimmed_report
    # reduce memory footprint
    del db
    gc.collect()
    return test_set_dict


def gen_export_dict(test_set_dict):
    '''
    export dict of form:
    { uuid_sd_days : {
        price : [],
        start_date : datetime.date,
        days : 90,
        post_code : "31691",
        geotag : [lat, long],
        brand : "SOME BRAND",
      }
    }
    '''
    features = {}
    export_dict = {}
    keys = list(test_set_dict.keys())
    obs = list(test_set_dict.values())
    for i, obs in enumerate(obs):
        key = keys[i]
        start_date = key.split('_')[1]
        year = int(start_date.split('-')[0])
        month = int(start_date.split('-')[1])
        day = int(start_date.split('-')[2])
        start_date = datetime.date(year=year, month=month, day=day)
        d = {}
        p = obs.get_daily_avg_price(start_date, 90)*100
        assert len(p) == 90
        d['price'] = p.round(1).tolist()
        d['start_date'] = str(start_date)
        d['days'] = 90
        d['post_code'] = obs.post_code
        d['geotag'] = obs.geotag
        d['brand'] = obs.brand.upper()
        export_dict[key] = d
        del test_set_dict[key]
        if i % 5000 == 0:
            gc.collect()
            print(
                f'  {i} of {len(keys)} observations processed')

    # reduce memory footprint
    del test_set_dict
    gc.collect()
    return export_dict


def main(dirs, start_date, end_date, sparsity_threshold, postal_region):
    database_directory = dirs['database_directory']
    price_window_filename = dirs['price_window_filename']
    major_q_range = pd.date_range(start_date, end_date, freq='QS-JAN')
    obs_counter = 0
    print('\nCollecting valid price window observations...')
    obs_counter, complete_obs_dict, db = parse_db_populate_obs_dict(dirs, postal_region, major_q_range, obs_counter)
    print(f'{obs_counter} observations found\n')

    print('Processing for export...')
    test_set_data_keys = list(complete_obs_dict.values())
    test_set_dict = generate_test_set_dict(db, test_set_data_keys, sparsity_threshold, postal_region)
    export_dict = gen_export_dict(test_set_dict)


    with open(os.path.join(
            database_directory, 
            f'{str(postal_region)}_{price_window_filename}'
        ), 'w') as f:
        export_json = json.dumps(export_dict, indent=4)
        f.write(export_json)
    print('All observations processed.\n')



if __name__ == '__main__':
    dirs = {
        'raw_german_data_folder' : 'tankerkoenig-data',
        'database_directory' : 'databases',
        'sta_info_filename' : 'sta_info_db',
        'price_db_filename' : 'price_sta_db.pkl',
    }
    start_date = datetime.date(2014, 10, 1)  # earliest Q start in data
    end_date = datetime.date(2023, 6, 1)  # will go up to end Q2 2023
    sparsity_threshold = 5  # one observation every X days, above thresh gets tossed
    postal_region = 0
    main(dirs, start_date, end_date, sparsity_threshold, postal_region)