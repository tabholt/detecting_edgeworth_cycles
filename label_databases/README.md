# Label Data Folder
The git repository contains the zipped / compressed labeled data for the replication package.

You must extract the JSON files from the zipped folder `label_data_files.zip`

## Steps to unzip
1. use your favorite unzipping program to extract the contents of the folder.
2. ensure that the files: 
   - `german_label_db.json`
   - `nsw_label_db.json`
   - `wa_label_db.json`

    are in the directory `detecting_edgeworth_cycles/label_databases/` (this directory)


## Detrended price windows data
Detrended price data is provided for all valid quarterly price windows from the German data from the period from Q4-2014 to Q4-2020 (inclusive). This data may be downloaded with the command:

`curl https://drive.switch.ch/index.php/s/yfFfuv6pNhGt7t6/download --output ALL_detrended_price_windows.json`

or by visiting https://drive.switch.ch/index.php/s/yfFfuv6pNhGt7t6/download in your web browser.