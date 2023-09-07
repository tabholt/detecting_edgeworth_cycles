'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

This class is the basic structure that matches a set of labels with a time series. 
The class should accomplish the following goals:
    - store the array of prices
    - produce array of timestamps corresponding to price array
    - store set of human labels corresponding to price array
      labels in {Yes, No, Maybe}
    - store the region {WA, NSW, GERMANY-{0..9}} the label is concerned with
    - store an identification key that can link to full station data
    - store information about any processing undergone by price array

this class needs to be universal for all of the datasets {WA, NSW, GERMANY}
'''
import csv
import json
import random
import datetime
import warnings
import numpy as np
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal


def load_label_set(label_db_path):
    with open(label_db_path, 'rb') as json_file:
        db = json.load(json_file)

    obj_db = Label_DB()
    for key in db:
        label_dict_obj = db[key]
        label_obj = Label()
        label_obj.add_dict_data(label_dict_obj)
        obj_db[key] = label_obj
    obj_db.set_region()
    obj_db.set_uuids()
    return obj_db


class Label_DB(dict):
    '''
    A custom dictionary-like class for managing and analyzing label data.

    This class extends the built-in Python dictionary and provides methods
    for loading, manipulating, and analyzing label data stored as Label objects.
    
    Attributes:
        region (str): The region associated with the labels in the database.
                      Can be one of {'germany', 'nsw', 'wa'}.
    
    Methods:
        set_uuids(self): Assigns UUIDs to all Label objects in the database.
        
        set_region(self): Sets the region attribute based on the first Label's region.
        
        get_train_test_sets(self, train_frac=1, shuffle=True, seed=42):
            Splits the Label objects into training and test sets.

        print_labels_statistics(self): Prints statistics about the distribution of labels.

        print_label_counts(self): Prints counts of different label types.

        plot_sample(self, n=10, price_type=None): Plots a sample of Label objects.

        export_to_csv(self, file_suffix): Exports the data to CSV files.

        export_sta_info_to_csv(self, filename): Exports station information to a CSV file.

    '''
    def set_uuids(self):
        for key in self.keys():
            self[key].uuid = str(key)

    def set_region(self):
        lbl = self[list(self.keys())[0]]
        self.region = lbl.region

    def get_train_test_sets(self, train_frac=1, shuffle=True, seed=42):
        '''
        will return maximum cardinality training and test sets.
        Variables:
            - train_frac : the fraction of the set to be used for training (0, 1]

        Returns:
            - train_set : a list of Label objects
            - test_set : a list of Label objects
        '''
        labels = []
        for key in self:
            lbl = self[key]
            labels.append(lbl)
        if shuffle:
            random.seed(seed)
            random.shuffle(labels)
        n = len(labels)
        n_train = int(n*(train_frac))
        if train_frac > 1:
            n_train = int(train_frac)
        train_set = labels[:n_train]
        test_set = labels[n_train:]
        return train_set, test_set

    def print_labels_statistics(self):
        print('=========================================')
        stats = defaultdict(int)
        labels = []
        for key in self:
            lbl = self[key]
            num_lbls = len(lbl.human_labels)
            stats[num_lbls] += 1
            labels.append(lbl.avg_rating)
        for key in [1, 2, 3, 4, 5]:
            print(
                f'{key} labeled obs = {stats[key]} of {len(self)} ({round(stats[key]/len(self)*100,1)}%)')
        print('=========================================')
        avg = np.mean(labels)
        print(f'average rating = {round(avg, 4)}')
        print('=========================================\n')

    def print_label_counts(self):
        print('=========================================')
        stats = defaultdict(int)
        labels = []
        for key in self:
            lbl = self[key]
            three_yes = lbl.three_yes
            stats['three_yes'] += lbl.three_yes
            stats['high_thresh'] += lbl.high_thresh
            stats['unanimous_no'] += lbl.unanimous_no
            stats['no_yeses'] += lbl.no_yeses
            stats['majority_no'] += lbl.majority_no
            stats['at_least_one_no'] += lbl.at_least_one_no
        for key in stats:
            print(
                f'{key} labeled obs = {stats[key]} of {len(self)} ({round(stats[key]/len(self)*100,1)}%)')
        print('=========================================')

    def plot_sample(self, n=10, price_type=None):
        '''
        price_type = {'price' : {WA, NSW},
                      'detrended_price' 
                     }
        '''
        keys = random.choices(list(self.keys()), k=n)
        for key in keys:
            lbl = self[key]
            lbl.plot(price_type=price_type)

    def export_to_csv(self, file_suffix):
        # db = self.get_cleaner_db(3, None)
        db = self
        f1 = open('1_' + file_suffix, 'w')
        f2 = open('2_' + file_suffix, 'w')
        w1 = csv.writer(f1)
        w2 = csv.writer(f2)
        max_labels = 0
        for key in db:
            sta_key = key[:key.index('_')]
            sta = db[key]
            if len(sta.human_labels) > max_labels:
                max_labels = len(sta.human_labels)
        w2_header = ['station_id', 'year', 'quarter']
        for i in range(max_labels):
            w2_header.append(f'label_{i+1}')
        w1.writerow(['station_id', 'year', 'quarter', 'day',
                    'retail_price', 'detrended_price'])
        w2.writerow(w2_header)
        for key in db:
            sta_key = key[:key.index('_')]
            sta = db[key]
            for day in range(sta.days):
                w1.writerow([
                    sta_key, sta.year, sta.quarter,  day+1,
                    sta.price[day], sta.detrended_price[day]
                ])
            h_labels = dict(enumerate(sta.human_labels))
            w2_row = [sta_key, sta.year, sta.quarter]
            for i in range(max_labels):
                w2_row.append(h_labels.get(i))
            w2.writerow(w2_row)
        f1.close()
        f2.close()

    def export_sta_info_to_csv(self, filename):
        f = open(filename, 'w')
        w = csv.writer(f)
        w.writerow([
            'station_id', 'brand', 'post_code', 'latitude', 'longitude'
        ])
        key_list = []
        for key in self:
            sta_key = key[:key.index('_')]
            sta = self[key]
            if sta_key not in key_list:
                w.writerow([
                    sta_key, sta.brand, sta.post_code, sta.geotag[0], sta.geotag[1]
                ])
                key_list.append(sta_key)


class Label(object):
    '''
    A class for representing label data associated with a specific observation.

    This class stores various attributes related to label data, such as human labels,
    price data, location information, and more, for a specific observation.

    Attributes:
        human_labels (numpy.ndarray): An array of human labels associated with the observation.
        price (numpy.ndarray): An array of price data.
        detrended_price (numpy.ndarray): An array of detrended price data.
        start_date (datetime.date): The start date of the observation.
        days (int): The number of days in the observation.
        post_code (str): The postal code associated with the observation location.
        geotag (tuple): A tuple of latitude and longitude coordinates (floats).
        region (str): The region associated with the observation location.
                     Can be one of {'germany', 'nsw', 'wa'}.
        brand (str): The brand associated with the observation.
        uuid (str): A unique identifier for the observation.

    Properties:
        default_data: Returns the default data (detrended_price).
        default_data_name: Returns the name of the default data ('detrended_price').
        units: Returns the units of the data ('cents/L').
        avg_rating: Returns the average human label rating.
        quarter: Returns the quarter of the year based on the start date.
        year: Returns the year based on the start date.
        delta_price: Returns the price differences between consecutive days.
        num_cycling_labels: Returns the count of labels indicating cycling events.
        num_non_cycling_labels: Returns the count of labels indicating non-cycling events.

    Methods:
        series_interface(self, series_name): Accesses different time series data.
        cycling_binary_interface(self, criterion): Computes cycling-related binary criteria.
        get_lombscargle(self): Calculates Lomb-Scargle periodogram.
        get_fft(self): Calculates Fast Fourier Transform.
        plot(self, price_type=None, verbosity=1): Plots the observation's data.
        plot_w_fft_lombscargle(self, price_type=None): Plots data with FFT and Lomb-Scargle.

    '''
    def __init__(self):
        self.human_labels = None  # np.ndarray of human labels {0, 0.5, 1}
        self.price = None  # np.ndarray
        self.detrended_price = None  # np.ndarray
        self.start_date = None  # datetime.date object
        self.days = 0  # int
        self.post_code = ''  # string
        self.geotag = (None, None)  # tuple of floats
        self.region = ''  # string in {'germany', 'nsw', 'wa'}
        self.brand = ''  # string
        self.uuid = ''  # string

    @property
    def default_data(self):
        return self.detrended_price

    @property
    def defualt_data_name(self):
        return 'detrended_price'

    @property
    def units(self):
        return 'cents/L'

    @property
    def avg_rating(self):
        return self.human_labels.mean()

    @property
    def quarter(self):
        return (self.start_date.month-1)//3 + 1

    @property
    def year(self):
        return self.start_date.year

    @property
    def delta_price(self):
        # 4x faster than np.diff()
        dp = self.price[1:] - self.price[:-1]
        return dp

    @property
    def num_cycling_labels(self):
        return np.count_nonzero(self.human_labels == 1)

    @property
    def num_non_cycling_labels(self):
        return np.count_nonzero(self.human_labels == 0)

    ##################################################
    # Cycling binaries
    #

    @property
    def two_yes(self):
        return self.num_cycling_labels >= 2

    @property
    def three_yes(self):
        return self.num_cycling_labels >= 3

    @property
    def unanimous_no(self):
        return self.num_non_cycling_labels == len(self.human_labels)

    @property
    def rounded(self):
        return bool(round(self.avg_rating))

    @property
    def no_yeses(self):
        return self.num_cycling_labels == 0

    @property
    def majority_no(self):
        return bool(round(self.num_non_cycling_labels/len(self.human_labels)))

    @property
    def at_least_one_no(self):
        return self.num_non_cycling_labels > 0

    @property
    def high_thresh(self):
        return self.avg_rating >= .8

    #
    # Cycling Binaries
    ##################################################

    def series_interface(self, series_name):
        mapper = {'price': self.price,
                  'detrended_price': self.detrended_price,
                  'delta_price': self.delta_price
                  }
        return mapper[series_name]

    def cycling_binary_interface(self, criterion):
        if criterion in ['two_yes', 'three_yes'] and len(self.human_labels) == 1:
            warnings.warn(
                'Not enough human labels for chosen truth criterion. Reverting to high_thresh.')
            criterion = 'high_thresh'
        mapper = {
            'two_yes': self.two_yes,
            'three_yes': self.three_yes,
            'rounded': self.rounded,
            'high_thresh': self.high_thresh,
            'majority_no': self.majority_no,
            'at_least_one_no': self.at_least_one_no,
            'unanimous_no': self.unanimous_no,
            'no_yeses': self.no_yeses,
        }
        return mapper[criterion]

    def __repr__(self):
        return f'{self.brand} @ {self.post_code} from {str(self.start_date)}'

    def add_dict_data(self, label_dict):
        self.human_labels = label_dict.get('human_labels')
        self.price = label_dict.get('price')
        self.detrended_price = label_dict.get('detrended_price')
        self.start_date = label_dict.get('start_date')
        self.days = label_dict.get('days')
        self.post_code = label_dict.get('post_code')
        self.geotag = label_dict.get('geotag')
        self.brand = label_dict.get('brand').upper()

        if self.human_labels != None:
            self.human_labels = np.array(self.human_labels)
        if self.price != None:
            self.price = np.array(self.price)
        if self.detrended_price != None:
            self.detrended_price = np.array(self.detrended_price)

        self.start_date = datetime.date(year=int(self.start_date[:4]), month=int(
            self.start_date[5:7]), day=int(self.start_date[8:]))

        if len(self.post_code) == 5:
            self.region = 'germany'
        elif int(self.post_code) in range(1000, 3000):
            self.region = 'nsw'
        elif int(self.post_code) in range(6000, 7000):
            self.region = 'wa'

    def get_lombscargle(self):
        series = self.detrended_price
        grid_resolution = 300  # set by experimentation.
        days = np.arange(0, len(series), 1)

        # calculate for periods that are in range of 2 to 90 days
        period_domain = np.linspace(2, 35, grid_resolution)
        ang_freqs = 2 * np.pi / period_domain
        # algorithm requires data to have 0 mean.
        series_0_mean = series - series.mean()
        if (series_0_mean == 0).all():
            lomb = np.full_like(period_domain, 0)
            return (period_domain, lomb, ([], []), 0)
        lomb = signal.lombscargle(
            days,
            series_0_mean,
            ang_freqs,
            normalize=True
        )
        thresh = lomb.max()/2  # discretionary value
        peaks, _ = signal.find_peaks(lomb, prominence=thresh)
        peaks_xy = (period_domain[peaks], lomb[peaks])
        lombscargle = (period_domain, lomb, peaks_xy, thresh)
        return lombscargle

    def get_fft(self):
        series = self.detrended_price
        days = len(series)
        fft = abs(np.fft.fft(series)/days)
        domain = np.arange(1, int(days/2))
        fft = fft[domain]  # original is a mirror image
        freq_domain = domain/days
        # remove 90 day period (series starts at 45 day period)
        fft = fft[1:]
        freq_domain = freq_domain[1:]
        thresh = fft.max()/2  # discretionary value
        peaks, _ = signal.find_peaks(fft, prominence=thresh)
        peaks_xy = (freq_domain[peaks], fft[peaks])  # peaks x and y values
        fourier = (freq_domain, fft, peaks_xy, thresh)
        return fourier

    def plot(self, price_type=None, verbosity=1):
        if price_type != None:
            p_array = self.series_interface(price_type)
        else:
            p_array = self.default_data
            price_type = self.defualt_data_name
        x_axis = np.arange(1, len(p_array) + 1)

        matplotlib.rc('figure', figsize=(9, 3.25))
        sns.set()  # set seaborn formatting
        fig, ax = plt.subplots(1, 1)
        plt.subplots_adjust(left=0.09, right=0.98,
                            bottom=.18, top=.83)

        ax.step(x_axis, p_array, linewidth=1,
                c=sns.color_palette()[0], label=price_type)
        ax.set_ylabel(price_type + f' ({self.units})')
        ax.set_xlabel('day')
        plt.xticks(rotation=60)
        title = price_type + \
            f'    Q{self.quarter} {self.year}     postcode:{self.post_code}    geocode:{str(self.geotag)}'
        ax.set_title(title)
        fig.suptitle(
            f'{self.region.upper()} - {self.brand} - station quarter observation')

        anno_choice = {1: 'Yes',  .5: 'Maybe', 0: 'No'}
        annotate = ''
        for i, rating in enumerate(self.human_labels):
            annotate += f'Labeler_{i} : {anno_choice[rating]}\n'
        plt.text(.01, .97, annotate, ha='left', va='top',
                 transform=ax.transAxes, size='large', c='red')

        plt.show()
        plt.close()

    def plot_w_fft_lombscargle(self, price_type=None):
        p_array = self.detrended_price
        price_type = 'detrended_price'

        fourier = self.get_fft()
        lombscargle = self.get_lombscargle()
        # fourier = (freq_domain, fft, peaks_xy, thresh)
        # lombscargle = (period_domain, lomb, peaks_xy, thresh)

        x_axis = np.arange(1, len(p_array) + 1)

        matplotlib.rc('figure', figsize=(7, 6))
        sns.set()  # set seaborn formatting
        fig, ax = plt.subplots(3, 1)
        plt.subplots_adjust(left=0.1, right=0.98,
                            bottom=.11, top=.87, hspace=.9)

        ax[0].step(x_axis, p_array, linewidth=1,
                   c=sns.color_palette()[0], label=price_type)
        ax[0].set_ylabel(price_type + f' ({self.units})')
        ax[0].set_xlabel('day')
        plt.sca(ax[0])
        plt.xticks(rotation=60)
        title = price_type + \
            f'    Q{self.quarter} {self.year}     postcode:{self.post_code}    geocode:{str(self.geotag)}'
        ax[0].set_title(title)
        fig.suptitle(
            f'{self.region.upper()} - {self.brand} - station quarter observation')

        plt.sca(ax[1])
        ax[1].plot(fourier[0], fourier[1], linewidth=1, c='red')
        ax[1].axhline(fourier[3], lw=1, c='lightgrey',
                      ls='--', label='thresh')
        ax[1].scatter(fourier[2][0], fourier[2][1],
                      marker='x', linewidth=2, c='orange')
        plt.legend(loc='upper right')
        # set the day labels on the peaks
        for i_x, i_y in zip(fourier[2][0], fourier[2][1]):
            plt.text(i_x+.002, i_y,
                     '{}days'.format(round(1/i_x, 1)), va='top')

        ax[1].set_title('descrete fourier transform - frequency domain')
        ax[1].set_xlabel('frequency (cycles/day)', labelpad=0)
        ax[1].set_ylabel('amplitute')
        ax[1].set_xlim(0, .5)
        # ax[1].set_ylim(-.05, 2.5)

        plt.sca(ax[2])
        ax[2].plot(lombscargle[0], lombscargle[1],
                   linewidth=1, c=sns.color_palette()[4])
        ax[2].axhline(lombscargle[3], lw=1, c='lightgrey',
                      ls='--', label='thresh')
        ax[2].scatter(lombscargle[2][0], lombscargle[2][1],
                      marker='x', linewidth=2, c=[sns.color_palette()[1]])
        plt.legend(loc='upper left')
        # set the day labels on the peaks
        for i_x, i_y in zip(lombscargle[2][0], lombscargle[2][1]):
            plt.text(i_x+.15, i_y,
                     '{}days'.format(round(i_x, 4)), va='top')

        ax[2].set_title('lomb-scargle normalized periodogram')
        ax[2].set_xlabel('period (days)', labelpad=0)
        ax[2].set_ylabel('lomb-scargle power')
        ax[2].set_xlim(0, lombscargle[0][-1])
        ax[2].set_ylim(-.05, 1.02)

        anno_choice = {1: 'Yes',  .5: 'Maybe', 0: 'No'}
        annotate = ''
        for i, rating in enumerate(self.human_labels):
            annotate += f'TA_{i} : {anno_choice[rating]}\n'
        plt.text(.01, .97, annotate, ha='left', va='top',
                 transform=ax[0].transAxes, size='large', c='red')

        plt.show()
        plt.close()
