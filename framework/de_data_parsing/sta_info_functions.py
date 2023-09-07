'''
Author: Timothy Holt - tabholt@gmail.com
Aug 2023

Script for processing and consolidating station information from Tankerkoenig 
data multiple CSV files (only station info, not prices) and exporting the data
as a pickled dictionary and a JSON file.

Get tankerkoenig data(~ 80 GB):
git clone https://tankerkoenig@dev.azure.com/tankerkoenig/tankerkoenig-data/_git/tankerkoenig-data

Parameters:
    - dirs (dict): A dictionary containing directory paths for various data
      sources. It should include the following keys:
        - 'raw_german_data_folder': The directory containing the raw CSV
          files.
        - 'database_directory': The directory where the output files will be
          saved.
        - 'sta_info_filename': The base filename for the output pickled and
          JSON files.
    - generate_sta_info_JSON (bool): If True, generates a JSON file alongside
      the pickled file.

Functions:
    - gen_file_list(dirs, sub_folder): Generates a sorted list of file paths
      from the specified sub-folder.
    - line_to_station(line): Converts a CSV line into a Station object.
    - info_file_to_dict(filename, sta_dict): Reads and parses a station info
      file and updates the station dictionary.
    - delete_sta_w_invalid_post_or_geotag(sta_dict): Removes stations with
      invalid postal codes or geotags from the dictionary.
    - main(dirs): The main function that orchestrates the processing and
      exporting of station information.

Note:
    - This script depends on external modules and classes defined in the
      'framework' package, including 'Timer_Utility.Timer' and
      'Station_Class.Station'.
    - Ensure that the required external modules and classes are properly
      imported.

Example Usage:
    - To run the script, provide the 'dirs' dictionary with appropriate
      directory paths and set any desired configuration options such as
      'generate_sta_info_JSON'.
    - Call the 'main' function with the 'dirs' dictionary as an argument to
      execute the script.
'''

import os
from os.path import join
import csv
import json
import pickle
from framework.de_data_parsing.Timer_Utility import Timer
from framework.de_data_parsing.Station_Class import Station

#################### SET PARAMETERS ####################
generate_sta_info_JSON = True
########################################################


def gen_file_list(dirs, sub_folder):
    raw_german_data_folder = dirs['raw_german_data_folder']
    file_walker = os.walk(join(raw_german_data_folder, sub_folder))
    file_list = []
    for item in file_walker:
        prefix = item[0]
        for name in item[2]:
            if 'DS_Store' in name:
                continue
            file_list.append(join(prefix, name))
    return sorted(file_list)


def line_to_station(line):
    sta = Station(
        uuid=line[0],
        name=line[1],
        brand=line[2],
        street=line[3],
        house_number=line[4],
        post_code=line[5],
        city=line[6],
        latitude=line[7],
        longitude=line[8]
    )
    return sta


def info_file_to_dict(filename, sta_dict):
    f = open(filename)
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        if i == 0:
            assert line[:9] == ['uuid', 'name', 'brand', 'street',
                                'house_number', 'post_code', 'city', 'latitude', 'longitude']
            continue
        if line[0] in sta_dict:
            continue
        station = line_to_station(line)
        sta_dict[station.uuid] = station
    f.close()
    return sta_dict


def delete_sta_w_invalid_post_or_geotag(sta_dict):
    # this function also sets proper datatype for geotag
    # note that post_codes cannot be stored as integers
    del_list = []
    for key, sta in sta_dict.items():
        try:
            _ = int(sta.post_code)
            geo = sta.geotag
            geo = (float(geo[0]), float(geo[1]))
            sta.geotag = geo
        except:
            del_list.append(key)
        if sta.geotag[0] == 0 or sta.geotag[1] == 0:
            if key not in del_list:
                del_list.append(key)
    for key in del_list:
        del sta_dict[key]
    return sta_dict

def main(dirs):
    database_directory = dirs['database_directory']
    sta_info_filename = dirs['sta_info_filename']
    t1 = Timer(string_reset=False, basic_string=True)
    # Generate list of files to process
    file_list = gen_file_list(dirs, 'stations')

    # Process info files
    sta_dict = {}
    print(f'\nprocessing {len(file_list)} station info files...')
    for i, filename in enumerate(file_list):
        if i % 200 == 0 and i > 0:
            print(f'    processed {i} of {len(file_list)} info files')
        sta_dict = info_file_to_dict(filename, sta_dict)
    print('processed all info files\n')

    print(f'number of stations found = {len(sta_dict)}')

    sta_dict = delete_sta_w_invalid_post_or_geotag(sta_dict)

    print(f'\nFinal number of stations = {len(sta_dict)}\n')

    #### EXPORT RESULTS ####
    if not os.path.exists(database_directory):
        os.makedirs(database_directory)
    with open(join(database_directory, f'{sta_info_filename}.pkl'), 'wb') as output:
        pickle.dump(sta_dict, output, pickle.HIGHEST_PROTOCOL)

    if generate_sta_info_JSON:
        json_dict = {}
        for key in sta_dict:
            json_dict[key] = sta_dict[key].__dict__

        export_json = json.dumps(json_dict, indent=4)
        f = open(join(database_directory, f'{sta_info_filename}.json'), 'w')
        f.write(export_json)
        f.close()

