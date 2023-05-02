'''
Author: Timothy Holt - tabholt@gmail.com
May 2022

Plots n random samples of data from target region 1 after the other
To Use:
    - call this script using python
    - provide command line arguments [region, n_samples]

    
Command Line Arguments:
    region : in {'de', 'nsw', 'wa'} - What region you would like 
    to plot a sample for

    n_samples : positive integer - how many samples to plot


Note:
    - Samples are chosen at random
    - You must close one plot before the next one will appear
    - Program ends when all plots have been closed
'''

from framework import Label_Class
import sys

db_locations = {
    'wa': 'label_databases/wa_label_db.json',
    'nsw': 'label_databases/nsw_label_db.json',
    'de': 'label_databases/german_label_db.json'
}

if len(sys.argv) < 3:
    raise Exception('proved argv = [region, n_samples]')
region = sys.argv[1]
n = int(sys.argv[2])
if region not in ['wa', 'nsw', 'de']:
    raise Exception('invalid region')


db = Label_Class.load_label_set(db_locations[region])
db.plot_sample(n)