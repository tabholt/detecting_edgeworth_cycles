'''
Author: Timothy Holt - tabholt@gmail.com
Ayg 2023

Class for representing gas station data and performing operations on it.

The Station class is designed to store and manipulate data related to gas stations,
including their unique identifiers, location information, and price data over time.

Attributes:
    - uuid (str): A unique identifier for the gas station.
    - name (str): The name of the gas station.
    - brand (str): The brand or franchise of the gas station.
    - street (str): The street where the gas station is located.
    - house_number (str): The house number or building number of the gas station.
    - post_code (str): The postal code of the gas station's location.
    - city (str): The city where the gas station is situated.
    - geotag (tuple): A tuple containing the latitude and longitude coordinates of
      the gas station's location.
    - price (list of floats): a list of all price observations reported for
      the station in chronological order (euros per liter).
    - timestamp (list of datetime.datetime): a list of the timestamps of all
      price observations reported for the station in chronological order. Can
      be zipped with price to form {timestamp: price} for each observation.

Methods:
    - start_date: Returns the start date of the price data.
    - end_date: Returns the end date of the price data.
    - max_price: Returns the maximum price and its corresponding timestamp.
    - delta_price: Returns the price differences between consecutive observations.
    - is_chronological: Checks if the price data is in chronological order.
    - num_quarters: Calculates the number of quarters covered by the data.
    - n: Returns the number of price observations.
    - days: Returns the total number of days covered by the data.
    - closest_terminal: Finds the closest wholesale terminal to the gas station.

Methods for Data Manipulation:
    - trim(begin, days=90): Creates a new Station object with data within a specified
      time window.
    - to_dataframe: Converts the gas station data to a Pandas DataFrame.
    - get_daily_price(s_date=None, days=None): Retrieves daily prices for a given
      date range.
    - get_hourly_price(s_date=None, days=None): Retrieves hourly prices for a given
      date range.
    - get_daily_avg_price(s_date=None, days=None): Calculates daily average prices
      for a given date range.
    - resample_daily(round_prices=True): Resamples the data to daily intervals,
      optionally rounding prices.

Utility Functions:
    - calc_distance_between(coord1, coord2): Calculates the distance between two
      GPS coordinates using the Haversine formula.

Example Usage:
    - Create a Station object, retrieve price data, perform data manipulations,
      and access station attributes.

Note:
    - The Station class is designed to work with gas station data, including
      price observations over time, and provides methods for data analysis and
      manipulation.

'''

# The Station class is a versatile container for gas station data and provides
# various methods for analyzing and manipulating this data. It is particularly
# useful for working with time series data related to gas station prices and
# locations.

# Example Usage:
# station = Station(
#     uuid="12345",
#     name="Gas Station A",
#     brand="Brand X",
#     street="123 Main St",
#     house_number="42",
#     post_code="12345",
#     city="Example City",
#     latitude=37.7749,
#     longitude=-122.4194
# )
# station.get_daily_avg_price()
# station.trim(datetime.date(2023, 1, 1), days=30)
# df = station.to_dataframe()


import datetime
import numpy as np
import pandas as pd
import math
import gc


class Station:
    def __init__(self,
                 uuid: str,
                 name: str,
                 brand: str,
                 street: str,
                 house_number: str,
                 post_code: str,
                 city: str,
                 latitude: float,
                 longitude: float
                 ):

        self.uuid = uuid
        self.name = name
        self.brand = brand
        self.street = street
        self.house_number = house_number
        self.post_code = post_code
        self.city = city
        self.geotag = (latitude, longitude)

        self.price = []  # float in euros per liter
        self.timestamp = []  # datatype = datetime.datetime

    @property
    def start_date(self):
        if type(self.timestamp[0]) == datetime.date:
            return self.timestamp[0]
        else:
            return self.timestamp[0].date()

    @property
    def end_date(self):
        if type(self.timestamp[0]) == datetime.date:
            return self.timestamp[-1]
        else:
            return self.timestamp[-1].date()

    @property
    def max_price(self):
        idx = np.argmax(self.price)
        return {self.timestamp[idx]: self.price[idx]}

    @property
    def delta_price(self):
        p = np.array(self.price)
        return p[1:] - p[:-1]


    @property
    def is_chronological(self):
        for i in range(self.n - 1):
            d = self.timestamp[i+1] - self.timestamp[i]
            d = d.seconds
            if d < 0:
                return False
        return True

    @property
    def num_quarters(self):
        dr = pd.date_range(self.start_date, self.end_date, freq='QS-JAN')
        if len(dr) == 0:
            return 0
        last_q_days = (self.end_date - dr[-1].date()).days
        if last_q_days < 90:
            dr = dr[:-1]
        return len(dr)

    @property
    def n(self):
        return len(self.price)

    @property
    def days(self):
        return (self.end_date-self.start_date).days

    @property
    def closest_terminal(self):
        # returns closest wholesale terminal
        terminals = {
            'North': (53.48, 9.95),  # refinery in Hamburg
            'Seefeld': (52.61, 13.68),  # tank farm in Seefeld
            'East': (52.44, 13.36),  # tank farm in Berlin
            'West': (51.54, 7.05),  # refinery in Essen
            'Southeast': (51.30, 12.02),  # refinery in Leuna
            'Rhine-Main': (50.12, 8.74),  # tank farm in Frankfurt
            'Southwest': (49.06, 8.34),  # refinery in Karlsruhe
            'South': (48.79, 11.47)  # refinery in Ingolstadt
        }
        min_dist = 99999  # km, initialize to big number
        closest_terminal = 'X'
        for key in terminals:
            d = self.calc_distance_between(self.geotag, terminals[key])
            if d < min_dist:
                min_dist = d
                closest_terminal = key
        return closest_terminal

    def __repr__(self):
        return f'{self.brand} @ {self.post_code} n={self.n}'

    def __str__(self):
        return f'{self.uuid} :\n    {self.name}\n    {self.brand}\n    {self.street} {self.house_number}\n    {self.post_code}\n    {self.city}\n    {self.geotag}\n'


    def trim(self, begin, days=90):
        '''
        Creates a Station object that contains all the
        same data as the parent Station with the time domain
        restricted to dates from (including) begin to begin + days
        In addition to all data from Stationreport, windows have:

            - unique uuid that is: 'original-uuid_start-date_days'

        Input dates must be formatted as dates, and not strings.

        Method returns the trimmed report, trimming not done in-place.
        '''

        # set assertions to facilitate error catching
        assert (
            # (type(begin) == pd._libs.tslibs.timestamps.Timestamp) or
            (type(begin) == datetime.date)
        )
        cutoff = begin + datetime.timedelta(days=days)
        begin_index = 0
        cutoff_index = -1
        # verify that begin and cutoff are valid dates, if so
        # change the indexes used to extract data
        if ((begin >= self.start_date) and (cutoff <= self.end_date)):
            for i in range(self.n):
                date = self.timestamp[i].date()
                if date == begin and begin_index == 0:
                    begin_index = i
                elif date > begin and begin_index == 0:
                    if i == 0:  # edge case - skip this one
                        break
                    begin_index = i - 1  # grab last used price before window date
                if date > cutoff:
                    cutoff_index = i - 1
                    break

        if cutoff_index <= begin_index:  # happens when time period has no reports
            # when there is a gap in reporting we will often see cutoff_index == begin_index
            return self

        trim_price = self.price[begin_index:cutoff_index]
        trim_timestamp = self.timestamp[begin_index:cutoff_index]

        # adjust the start date so that it aligns with the window
        if trim_timestamp[0].date() > begin:  # WILL BE BUGGY FOR NON-TIMESTAMPS
            if type(trim_timestamp[0]) == datetime.date:
                trim_timestamp[0] = begin
            else:
                trim_timestamp[0] = datetime.datetime.combine(
                    begin, datetime.datetime.min.time())  # datetime.datetime format
        if trim_timestamp[-1].date() < cutoff:
            if type(trim_timestamp[0]) == datetime.date:
                trim_timestamp.append(cutoff)
            else:
                trim_timestamp.append(datetime.datetime.combine(
                    cutoff, datetime.datetime.min.time()))  # datetime.datetime format
            trim_price.append(trim_price[-1])

        # create dictionary for trimmed report using begin and cutoff
        # indexes to select sub-arrays out of the original data arrays
        trimmed_report = Station(
            uuid=f'{self.uuid}_{str(begin)}_{days}',
            name=self.name,
            brand=self.brand,
            street=self.street,
            house_number=self.house_number,
            post_code=self.post_code,
            city=self.city,
            latitude=self.geotag[0],
            longitude=self.geotag[1]
        )

        trimmed_report.price = trim_price
        trimmed_report.timestamp = trim_timestamp

        return trimmed_report

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for attr in ['uuid', 'name', 'brand', 'street', 'house_number', 'post_code', 'city']:
            col = [getattr(self, attr)] * self.n
            df[attr] = col
        df['latitude'] = [self.geotag[0]] * self.n
        df['longitude'] = [self.geotag[1]] * self.n
        df['timestamp'] = self.timestamp
        df['price'] = self.price
        return df

    def get_daily_price(self, s_date=None, days=None):
        '''
        return: a vector that has the active End of Day (EoD)
        price for every day in the requested range.

        How it works:
            - if requested range is not compatable with the given
              station report then returns vector of np.nan

            - generates an empty vector of length days.

            - fills first entry of vector with last observed price
              before start date (if no price data on s_date)

            - fills the appropriate day on the vector with the EoD
              price update of each day where there is a price update

            - fills in empty spots of vector (ie where there was no
              price update) with the last price observed.
        '''
        if s_date is None:
            s_date = self.start_date
            days = (self.end_date-self.start_date).days

        assert type(s_date) == datetime.date
        e_date = s_date + datetime.timedelta(days=days)
        if s_date < self.start_date or e_date > self.end_date:
            print('ciao')
            return np.full(days, np.nan, dtype='float64')

        price_vec = self.price

        s_date = pd.to_datetime(s_date)
        daily_price = np.full(days, np.nan, dtype='float64')
        for i, ts in enumerate(self.timestamp):
            day = (ts - s_date).days
            if day < 0:
                daily_price[0] = price_vec[i]
            elif day < days:
                daily_price[day] = price_vec[i]
            else:
                break

        for i in range(len(daily_price)):
            if np.isnan(daily_price[i]):
                # works because daily_price[0] is garanteed to have value.
                daily_price[i] = daily_price[i-1]

        return daily_price

    def get_hourly_price(self, s_date=None, days=None):
        '''
        return: a vector that has the active End of Hour (EoH)
        price for every hour in the requested range.

        How it works:
            - if requested range is not compatable with the given
              station report then returns vector of np.nan

            - generates an empty vector of length days*24.

            - fills first entry of vector with last observed price
              before start hour (if no price data on first hour)
              or with the first observed price in situations where
              there is no price update before H_0

            - fills the appropriate day on the vector with the EoH
              price update of each hour where there is a price update

            - fills in empty spots of vector (ie where there was no
              price update) with the last price observed.
        '''
        if s_date is None:
            s_date = self.start_date
            days = (self.end_date-self.start_date).days
        hours = days*24

        assert type(s_date) == datetime.date
        e_date = s_date + datetime.timedelta(days=days)
        if s_date < self.start_date or e_date > self.end_date:
            return np.full(hours, np.nan, dtype='float64')

        price_vec = self.price

        s_date = pd.to_datetime(s_date)
        hourly_price = np.full(hours, np.nan, dtype='float64')
        for i, ts in enumerate(self.timestamp):
            hour = int((ts - s_date).total_seconds()/3600)
            if hour < 0:
                hourly_price[0] = price_vec[i]
            elif hour < hours:
                hourly_price[hour] = price_vec[i]
            else:
                break

        # there exists edge case where s_date is day one, but price
        # begins later in day. In this case we backfill the front of
        # the array with the price from first update.
        if np.isnan(hourly_price[0]):
            i = 0
            while np.isnan(hourly_price[i]):
                i += 1
            hourly_price[0] = hourly_price[i]

        for i in range(len(hourly_price)):
            if np.isnan(hourly_price[i]):
                # works because hourly_price[0] is garanteed to have value.
                hourly_price[i] = hourly_price[i-1]

        return hourly_price

    def get_daily_avg_price(self, s_date=None, days=None):
        hourly_price = self.get_hourly_price(s_date, days)
        assert len(hourly_price) % 24 == 0
        days = int(len(hourly_price)/24)
        daily_avg_price = np.ndarray(days)
        for i in range(days):
            start_index = i*24
            end_index = start_index + 24
            day_prices = hourly_price[start_index:end_index]
            daily_avg_price[i] = day_prices.mean()
        return daily_avg_price
    
    def resample_daily(self, round_prices=True):
        prices = self.get_daily_avg_price()
        if round_prices:
            prices = prices.round(3)
        self.prices = prices.tolist()
        dates = pd.date_range(start=self.start_date, end=self.end_date, inclusive='left')
        self.timestamp = [datetime.datetime(d.year,d.month,d.day) for d in dates]


    def calc_distance_between(self, coord1, coord2):
        '''
        uses haversine formula to calculate distances between
        two GPS coordinates
        requires as inputs:
            coord1 = (latitude1, longitude1)
            coord2 = (latitude2, longitude2)
            lats and longs must be in signed decimal degrees

        Returns:
            Distance between two points IN KILOMETERS
        '''
        lat1, lon1 = coord1[0], coord1[1]
        lat2, lon2 = coord2[0], coord2[1]

        R = 6371e3  # radius of earth in meters
        phi1 = lat1 * math.pi / 180  # phi and lambda in radians
        phi2 = lat2 * math.pi / 180
        delta_phi = phi2 - phi1
        delta_lambda = (lon2 - lon1) * math.pi / 180

        a = (
            (math.sin(delta_phi/2))**2 +
            math.cos(phi1) * math.cos(phi2) *
            (math.sin(delta_lambda/2))**2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        distance = R * c  # in meters

        return round(distance / 1000, 1)  # in km


def sta_dict_to_dataframe(sta_dict):
    df_list = [sta.to_dataframe() for sta in sta_dict.values()]
    df = pd.concat(df_list, ignore_index=True)
    return df

def sta_dict_resample_daily(sta_dict):
    for v in sta_dict.values():
        v.resample_daily()
    gc.collect()
    return sta_dict

def dataframe_to_sta_dict(df):
    sta_dict = {}
    keys = df['uuid'].to_list()
    prices = df['price'].to_numpy()
    key_set = df['uuid'].unique()
    indices = []
    start_ptr = 0
    for key in key_set:
        idx = keys.index(key, start_ptr)
        indices.append(idx)
        start_ptr = idx
    indices.append(len(keys))
    for i, key in enumerate(key_set):
        l = indices[i]
        r = indices[i+1]
        d = df.iloc[l]
        sta = Station(
            uuid=d['uuid'],
            name=d['name'],
            brand=d['brand'],
            street=d['street'],
            house_number=d['house_number'],
            post_code=d['post_code'],
            city=d['city'],
            latitude=d['latitude'],
            longitude=d['longitude'],
        )
        sta.price = prices[l:r].tolist()
        sta.timestamp = df['timestamp'][l:r].tolist()
        sta_dict[d.uuid] = sta
    return sta_dict
