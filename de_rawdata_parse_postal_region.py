'''
Author: Timothy Holt - tabholt@gmail.com
Sep 2023

Parse the raw tankerkoenig german gasoline price data into convenient 
data structures.
Requires:
    Tankerkoenig raw data, to get raw data use:
    git clone https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data
    The raw data folder will require about 80GB of disk space.
    The raw data should be in the folder: 
    framework/de_data_parsing/parameters.py dir['raw_german_data_folder']

    
To Use:
    - download tankerkoenig raw data
    - call this script using python
    - provide command line argument [postal_region]

    
Command Line Arguments:
    postal_region : in {0..9, all} - German postal region to process. Passing
        the argument 'all' will parse all regions from 0 to 9.

        
Note:
    - Parsing the entire tankerkoenig data set will take several hours.
    - At least 8GB of RAM memory will be necessary to parse the dataset. 16GB 
      or more is recommended. 
    - Data is partitioned into "Postal regions" for the parsing due to memory
      and performance constraints.
    - Data is usable either from the price window format or the Station class
      format that will be saved in the data output folder. The Station class is
      defined in framework/de_data_parsing/Station_Class.py and should be very
      useful to advanced users. The data structure contains all price 
      observations sorted by station in the Station objects.
    - Raw data contains only raw price data, detrending this data to compensate
      for fluctuations in underlying commodities prices requires additional data.
    - Json price windows output files may be converted to csv format using the
      script convert_price_window_json_csv.py

      
Output:
    - price_sta_db.pkl files: dictionaries of form {'station_uuid': Station_Object}
      for all stations in the raw data. Files are divided by postal region.
    - price_windows.json files: raw data parsed into quarterly daily average
      price observation windows as was used in paper.
'''

import os
from os.path import join
import sys
import gc
import datetime
from framework.de_data_parsing.Timer_Utility import Timer
import framework.de_data_parsing.sta_info_functions as sta_info_functions
import framework.de_data_parsing.price_db_functions as price_db_functions
import framework.de_data_parsing.observation_windows_functions as observation_windows_functions
from framework.de_data_parsing.parameters import *


def load_argv():
    try:
        textin = sys.argv[1]
        if textin.lower() in 'all':
            return list(range(10))
        assert int(textin) in range(10)
    except:
        print(
            'Usage: argv = [postal_region]\n    postal_region in {0..9, all}\n')
        exit()
    return [int(textin)]

def test_if_already_exists(region_list):
    already_exists = []
    for postal_region in region_list:
        fname1 = join(
            dirs['database_directory'], 
            f"{postal_region}_{dirs['price_db_filename']}")
        fname2 = join(
            dirs['database_directory'], 
            f"{postal_region}_{dirs['price_window_filename']}")
        if os.path.exists(fname1) and os.path.exists(fname2):
            already_exists.append(postal_region)
    if already_exists:
        while True:
            overwrite = input(f'regions {already_exists} have already been processed. Overwrite? (y/n): ')
            if overwrite.lower() in {'y','n','yes','no'}:
                break
        if overwrite.lower() in 'no':
            return [r for r in region_list if r not in already_exists]
    return region_list

region_list = load_argv()
region_list = test_if_already_exists(region_list)
print(f'Postal regions to process: {region_list}\n')
t1 = Timer(f'Total Processing Time')
for postal_region in region_list:
    t2 = Timer(f'Postal Region {postal_region}')
    if not os.path.exists(join(dirs['database_directory'], f"{dirs['sta_info_filename']}.pkl")):
        print('PARSING STATION INFO')
        sta_info_functions.main(dirs)
        gc.collect()
    t2.save_and_reset('Parse station info data')

    price_db_functions.main(dirs, postal_region)
    gc.collect()
    t2.save_and_reset('Parse station price data')

    observation_windows_functions.main(dirs, start_date, end_date, sparsity_threshold, postal_region)
    t2.save_and_reset('Generate observation windows')
    t2.print_all_timers()
    t1.save_and_reset(f'Postal region {postal_region}')
t1.print_all_timers()