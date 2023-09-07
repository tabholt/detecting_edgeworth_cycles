'''
Author: Timothy Holt - tabholt@gmail.com
Aug 2023

Set parameters for processing gas station data.

This parameter set defines directory paths and filenames for storing and accessing
gas station data, as well as parameters related to a 90-day time window.

Parameters:
    - dirs (dict): A dictionary containing directory and filename settings.
        - 'raw_german_data_folder' (str): The folder for raw German data.
        - 'database_directory' (str): The directory for storing databases.
        - 'sta_info_filename' (str): The filename for station information database.
        - 'price_db_filename' (str): The filename for the price station database.
        - 'price_window_filename' (str): The filename for price windows in JSON format.
        - 'price_window_csv_filename' (str): The filename for price windows in CSV format.

    - start_date (datetime.date): The start date for the 90-day window, set to the
      earliest quarter start date in the data.

    - end_date (datetime.date): The end date for the 90-day window, set to go up
      to the end of Q2 2023.

    - sparsity_threshold (int): The threshold for data sparsity, determining the
      minimum number of days between observations to keep them.

Note:
    - These parameters are used in data processing and analysis for gas station data.
    - The 'dirs' dictionary defines paths and filenames for data storage and retrieval.
    - The 'start_date' and 'end_date' specify the time window of interest.
    - The 'sparsity_threshold' controls the data filtering based on observation frequency.
'''

# Example Usage:
# Set up the parameters for gas station data processing.
# dirs = {
#     'raw_german_data_folder': 'tankerkoenig-data',
#     'database_directory': 'de_databases',
#     'sta_info_filename': 'sta_info_db',
#     'price_db_filename': 'price_sta_db.pkl',
#     'price_window_filename': 'price_windows.json',
#     'price_window_csv_filename': 'price_windows.csv'
# }
# start_date = datetime.date(2014, 10, 1)
# end_date = datetime.date(2023, 6, 1)
# sparsity_threshold = 5

import datetime

########################################################
#################### SET PARAMETERS ####################
########################################################

### DIRECTORIES AND FILENAMES
dirs = {
    'raw_german_data_folder' : 'tankerkoenig-data',
    'database_directory' : 'de_databases',
    'sta_info_filename' : 'sta_info_db',
    'price_db_filename' : 'price_sta_db.pkl',
    'price_window_filename' : 'price_windows.json',
    'price_window_csv_filename' : 'price_windows.csv'
}

### 90 day window parameters
start_date = datetime.date(2014, 10, 1)  # earliest Q start in data
end_date = datetime.date(2023, 6, 1)  # will go up to end Q2 2023
sparsity_threshold = 5  # one observation every X days, above thresh gets tossed
########################################################