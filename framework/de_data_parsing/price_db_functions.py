'''
Author : Timothy Holt
Date : Aug 2023


Functions to be used for processing and consolidating Tankerkoenig fuel price 
information from multiple CSV files and exporting the data as a pickled 
database.


Get tankerkoenig data(~ 80 GB):
git clone https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data


Parameters:
    - dirs (dict): A dictionary containing directory paths for various data
      sources. It should include the following keys:
        - 'database_directory': The directory where the output database will
          be saved.
        - 'price_db_filename': The base filename for the output pickled
          database.
    - postal_region (int): The postal region code for which to generate the
      fuel price database.


Global Constants:
    - reject_near_zero_price_obs (bool): If True, rejects price observations
      below a specified threshold.
    - reject_if_delta_P_threshold (float): The threshold for rejecting price
      observations based on price changes.
    - min_days (int): Minimum number of days a station must report prices to
      be included in the database.
    - resample_to_daily (bool): If True, resamples prices to daily averages.


Functions:
    - load_argv(): Loads the postal region code from command-line arguments.
    - load_sta_info_db(dirs): Loads the station information database
      previously generated.
    - export_prices_db(dirs, prices_db, postal_region): Exports the generated
      fuel price database.
    - add_file_to_db(filename, sta_db): Adds price data from a file to the
      station database.
    - delete_irrelevant_postcodes(sta_db, postal_region): Deletes stations
      with postcodes not matching the postal region.
    - remove_empty_stations(sta_db): Removes stations with no price data.
    - remove_stations_with_few_days(sta_db, min_days): Removes stations with
      fewer reporting days.
    - gen_prices_db(dirs, postal_region): Generates the fuel price database
      for the specified postal region.
    - main(dirs, postal_region): The main function that orchestrates the
      processing and exporting of fuel price data.


Note:
    - This script depends on external modules and classes defined in the
      'framework' package, including 'sta_info_functions.gen_file_list' and
      the 'Station' class.
    - Ensure that the required external modules and classes are properly
      imported.

```

This docstring provides an overview of the script's purpose, parameters, global constants, and functions. It also mentions the external dependencies and provides an example usage guide.
'''
import os
import sys
import pickle
import datetime
from framework.de_data_parsing.sta_info_functions import gen_file_list

#################### SET PARAMETERS ####################
reject_near_zero_price_obs = True
reject_if_delta_P_threshold = .5  # euros change between 2 observations
min_days = 90 # delete stations with fewer than this number of days reporting
resample_to_daily = False # if True, prices will be resampled to daily average
########################################################

def load_argv():
    try:
        postal_region = sys.argv[1]
        assert int(postal_region) in range(10)
    except:
        print(
            'Usage: argv = [postal_region]\n    postal_region in {0..9}\n')
        exit()
    return postal_region


def load_sta_info_db(dirs):
    database_directory = dirs['database_directory']
    sta_info_filename = dirs['sta_info_filename']
    with open(os.path.join(database_directory, f'{sta_info_filename}.pkl'), 'rb') as pickle_file:
        sta_db = pickle.load(pickle_file)
    return sta_db


def export_prices_db(dirs, prices_db, postal_region):
    database_directory = dirs['database_directory']
    price_db_filename = dirs['price_db_filename']
    db_filename = f'{str(postal_region)}_{price_db_filename}'
    if not os.path.exists(database_directory):
        os.makedirs(database_directory)
    with open(os.path.join(database_directory, db_filename), 'wb') as output:
        pickle.dump(prices_db, output, pickle.HIGHEST_PROTOCOL)


def add_file_to_db(filename, sta_db):
    '''
    line = [date, station_uuid, diesel, e5, e10, dieselchange, e5change, e10change]

        Because there are no complicated text fields with commas, no need to
        use a csv parser. Simply break the text strings at each comma. 30-50%
        faster than csv parser.
    '''
    f = open(filename)
    for i, line in enumerate(f):
        line = line.rstrip('\n')
        line = line.split(',')
        if i == 0:
            assert line == ['date', 'station_uuid', 'diesel',
                            'e5', 'e10', 'dieselchange', 'e5change', 'e10change']
            continue
        if line[6] == '0':  # e5 price didn't change
            continue
        uuid = line[1]
        if uuid not in sta_db:
            continue
        timestamp = datetime.datetime.strptime(
            line[0][:16], '%Y-%m-%d %H:%M')
        price = float(line[3])  # line[3] corresponds to e5 prices
        if reject_near_zero_price_obs:
            if price < 0.2:  # Prices of less than 20 cents per liter must be erroneous
                continue
        if sta_db[uuid].price != []:
            if abs(price-sta_db[uuid].price[-1]) > reject_if_delta_P_threshold:
                continue
        sta_db[uuid].price.append(price)
        sta_db[uuid].timestamp.append(timestamp)
    f.close()
    return sta_db



def delete_irrelevant_postcodes(sta_db, postal_region):
    del_list = []
    for key in sta_db:
        if sta_db[key].post_code[0] != str(postal_region):
            del_list.append(key)
    for key in del_list:
        del sta_db[key]
    return sta_db


def remove_empty_stations(sta_db):
    keys = list(sta_db.keys())
    for key in keys:
        if sta_db[key].n == 0:
            del sta_db[key]
    return sta_db

def remove_stations_with_few_days(sta_db, min_days=90):
    keys = list(sta_db.keys())
    for key in keys:
        if sta_db[key].days < min_days:
            del sta_db[key]
    return sta_db


def gen_prices_db(dirs, postal_region):

    # Generate list of files to process
    file_list = gen_file_list(dirs, 'prices')

    # load info database
    sta_db = load_sta_info_db(dirs)
    sta_db = delete_irrelevant_postcodes(sta_db, postal_region)

    # Process files
    for i, filename in enumerate(file_list):
        if i % 500 == 0 and i > 0:
            print(
                f'    processed {i} of {len(file_list)} price files')
        sta_db = add_file_to_db(filename, sta_db)
    return sta_db


def main(dirs, postal_region):
    print(f'Processing price files for region {postal_region}')
    prices_db = gen_prices_db(dirs, postal_region)
    prices_db = remove_empty_stations(prices_db)
    prices_db = remove_stations_with_few_days(prices_db, min_days=min_days)
    print(f"Processed all price files. Exporting price db...")
    export_prices_db(dirs, prices_db, postal_region)
    del prices_db


if __name__ == '__main__':
    postal_region = load_argv()
    main(postal_region)
