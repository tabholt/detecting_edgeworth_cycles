'''
Author: Timothy Holt - tabholt@gmail.com
May 2023

This file contains:
    - The mathematical basis for the parametric models
    - The Memoized Adaptive Grid Search (MAGS) algorithm
    - The Model class which is the interface for parametric models
      and is also used to extract the features for the machine
      learning methods.

Notes:
    - All methods are treated with a uniform interface
    - The b_ functions contain the mathematical definition of models
    - The g_ functions are boolean versions to evaluate cycling status
    - Model class is instantiated with a list of Label objects as defined
      in file Label_Class.py
'''

import math
import time
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from collections import defaultdict
from hashlib import md5


def b_PRNR(delta_price):
    '''
    *** Positive Runs vs. Negative Runs PRNR ***
     - as per Castanias and Johnson 0 changes are
       considered positive
     - delta_price must be np.array
    '''
    runs = np.where(delta_price >= 0, 1, 0)
    pos = []
    neg = []
    count = 0
    for i in range(len(runs)-1):
        count += 1
        if runs[i+1] != runs[i]:
            if runs[i] == 0:
                neg.append(count)
            else:
                pos.append(count)
            count = 0
    if runs[-1] == 0:
        neg.append(count+1)
    else:
        pos.append(count+1)
    mean_pos = 0
    mean_neg = 0
    if len(pos) > 0:
        mean_pos = np.mean(pos)
    if len(neg) > 0:
        mean_neg = abs(np.mean(neg))
    return mean_pos - mean_neg


def b_NMC(delta_price):
    '''
    *** Negative Median Change NMC ***
    '''
    return np.median(delta_price)


def b_MIMD(delta_price):
    '''
    *** Mean Increase vs. Mean Decrease MIMD ***
     - delta_price must be np.array
    '''
    pos = delta_price[delta_price > 0]
    neg = delta_price[delta_price < 0]
    mean_pos = 0
    mean_neg = 0
    if len(pos) > 0:
        mean_pos = np.mean(pos)
    if len(neg) > 0:
        mean_neg = abs(np.mean(neg))
    return mean_pos - mean_neg


def b_MBPI(delta_price):
    '''
    *** Many Big Price Increases MBPI ***
     - no computation can be done before
       parameters are known
     - see g func for computational details
    '''
    return delta_price


def b_WAVY(detrended_price):
    '''
    *** Waviness WAVY ***
    '''
    w = np.abs(detrended_price - np.mean(detrended_price))
    return np.sum(w)


def b_FT0(detrended_price):
    '''
    *** Fourier Transform - max FT0 ***
    '''
    fft, _ = get_fft(detrended_price)
    max_fft = np.max(fft)
    return max_fft


def b_FT1(detrended_price):
    '''
    *** Fourier Transform - peak height FT1 ***
    '''
    _, peak_heights = get_fft(detrended_price)
    if len(peak_heights) == 0:
        max_peak_height = 0
    else:
        max_peak_height = np.max(peak_heights)
    return max_peak_height


def b_FT2(detrended_price):
    '''
    *** Fourier Transform - HHI FT2 ***
    '''
    fft, _ = get_fft(detrended_price)
    hhi = np.sum(np.multiply(fft, fft))
    return hhi


def b_LS0(detrended_price):
    '''
    *** LS Periodogram - max LS0 ***
    '''
    lomb, _ = get_ls(detrended_price)
    max_lomb = np.max(lomb)
    return max_lomb


def b_LS1(detrended_price):
    '''
    *** LS Periodogram - peak height LS1 ***
    '''
    _, peak_heights = get_ls(detrended_price)
    if len(peak_heights) == 0:
        max_peak_height = 0
    else:
        max_peak_height = np.max(peak_heights)
    return max_peak_height


def b_LS2(detrended_price):
    '''
    *** LS Periodogram - HHI LS2 ***
    '''
    lomb, _ = get_ls(detrended_price)
    hhi = np.sum(np.multiply(lomb, lomb))
    return hhi


def b_CS0(detrended_price):
    '''
    *** Cubic Spline - Root CS0 ***
    '''
    y = detrended_price - np.mean(detrended_price)
    x = np.arange(0, len(y), 1)
    cs = CubicSpline(x, y)
    return len(cs.roots())


def b_CS1(detrended_price):
    '''
    *** Cubic Spline - Integral CS1 ***
    '''
    y = detrended_price - np.min(detrended_price)
    x = np.arange(0, len(y), 1)
    cs = CubicSpline(x, y)
    return cs.integrate(x[0], x[-1])


def get_ls(detrended_price, domain=None, grid_resolution=300):
    normalize = False
    if domain == None:
        domain = (2, 45)
    days = np.arange(0, len(detrended_price), 1)
    # calculate for periods that are in domain
    period_domain = np.linspace(domain[0], domain[1], grid_resolution)
    ang_freqs = 2 * np.pi / period_domain
    # algorithm requires data to have 0 mean.
    series_0_mean = detrended_price - detrended_price.mean()
    if (series_0_mean == 0).all():
        lomb = np.full_like(period_domain, 0)
    else:
        lomb = signal.lombscargle(
            days,
            series_0_mean,
            ang_freqs,
            normalize=normalize
        )
    if not normalize:
        lomb /= 1000  # SCALING FACTOR
    thresh = lomb.max()/2  # discretionary value
    peaks, _ = signal.find_peaks(lomb, prominence=0)
    peak_heights = lomb[peaks]
    return lomb, peak_heights


def get_fft(detrended_price):
    days = len(detrended_price)
    fft = abs(np.fft.fft(detrended_price)/days)
    domain = np.arange(1, int(days/2))
    fft = fft[domain]  # original is a mirror image
    # remove 90 day period (series starts at 45 day period)
    freq_domain = domain/days
    fft = fft[1:]
    fft /= 10  # SCALING FACTOR
    freq_domain = freq_domain[1:]
    thresh = fft.max()/2  # discretionary value
    peaks, _ = signal.find_peaks(fft, prominence=0)
    peak_heights = fft[peaks]
    return fft, peak_heights


def Lambda(g_array):
    g = np.where(g_array > 709, 709, g_array)  # numerical maximum exp
    return np.exp(g)/(1+np.exp(g))


def log_l(truth_array, Lambda_array):
    L = np.where(np.isclose(Lambda_array, 1), .999999, Lambda_array)
    ll_array = truth_array * np.log(L) + (1 - truth_array) * np.log(1-L)
    return np.sum(ll_array)


class Model(object):
    '''
    Model for handling labeled observations and feature evaluation.

    This class represents a model for working with labeled observations. It
    provides methods for evaluating features, calculating various metrics,
    and managing observations.

    Args:
        obs_set (list of Label): List of labeled observations.
        truth_criterion (str): The truth criterion for labeling observations.
        region (str, optional): The region associated with the dataset.

    Properties:
        obs_set_hash: Returns the hash identifier for the observation set.
        uuid_list: Returns a list of UUIDs of observations.
        n: Returns the number of observations in the set.
        detrended_price_array: Returns a 2D array of detrended prices.
        delta_price_array: Returns a 2D array of delta prices.
        LS_array: Returns a 2D array of Lomb-Scargle periodograms.
        MBPI_theta1_star: Returns theta1_star for MBPI method.
        method_name_dict: Returns a dictionary of method names.
        method_eval_dict: Returns a dictionary of method evaluation functions.
        method_g_dict: Returns a dictionary of method g-functions.
        method_base_dict: Returns a dictionary of base methods.

    Methods:
        calc_method_array: Calculates evaluation arrays for given methods.
        evaluate_external_data: Evaluates methods on external data.
        get_truth_array: Retrieves ground truth labels.
        setup_LL_func: Sets up a log-likelihood function for optimization.
        MAGS: Memoized Adaptive Grid Search for optimization.
        evaluate_and_print_method: Evaluates and prints method performance.
        evaluate_theta: Evaluates a method's performance for a given theta.
        eval_and_print_LL_func: Evaluates and prints log-likelihood functions.
        print_truth_table: Prints a truth table summarizing model performance.
        evaluate_truth: Evaluates the truth against evaluations.
        Various g-functions and evaluation methods for specific methods.
    '''
    def __init__(self, obs_set, truth_criterion, region=''):
        self.obs_set = obs_set  # list of Label objects
        self.truth_criterion = truth_criterion  # string
        self.region = region  # string in {'de', 'nsw', 'wa}
        self.truth = self.get_truth_array()  # np array containing truth bools
        self.method_arrays = {}  # dictionary containing np.arrays of b_func evals

    @property
    def obs_set_hash(self):
        tr_uuids = [obs.uuid for obs in self.obs_set]
        h = md5()
        for uuid in sorted(tr_uuids):
            h.update(uuid.encode())
        return h.hexdigest()

    @property
    def uuid_list(self):
        uuid_list = []
        for ob in self.obs_set:
            uuid_list.append(ob.uuid)
        return uuid_list

    @property
    def n(self):
        return len(self.obs_set)

    @property
    def detrended_price_array(self):
        detrended_price = np.ndarray(shape=(self.n, 90))
        for i, ob in enumerate(self.obs_set):
            detrended_price[i] = ob.detrended_price
        return detrended_price

    @property
    def delta_price_array(self):
        delta_price = np.ndarray(shape=(self.n, 89))
        for i, ob in enumerate(self.obs_set):
            delta_price[i] = ob.delta_price
        return delta_price

    @property
    def LS_array(self):
        grid_resolution = 90
        LS = np.ndarray(shape=(self.n, 90))
        for i, ob in enumerate(self.obs_set):
            LS[i], _ = get_ls(ob.detrended_price,
                              grid_resolution=grid_resolution)
        return LS

    def LS_array_external_data(self, data):  # data should be detrended price
        grid_resolution = 90
        LS = np.ndarray(shape=(data.shape[0], 90))
        for i in range(data.shape[0]):
            LS[i], _ = get_ls(data[i, :], grid_resolution=grid_resolution)
        return LS

    @property
    def MBPI_theta1_star(self):
        theta1_star, _, _ = self.MAGS('MBPI', 0, 50, 2, 50, verbose=0)
        return theta1_star

    @property
    def method_name_dict(self):
        return {'PRNR': 'Positive Runs vs. Negative Runs',
                'NMC': 'Negative Median Change',
                'MIMD': 'Mean Increase vs. Mean Decrease',
                'MBPI': 'Many Big Price Increases',
                'WAVY': 'Waviness',
                'FT0': 'Fourier Transform Simple Max',
                'FT1': 'Fourier Transform Tallest Peak',
                'FT2': 'Fourier Transform HH Index',
                'LS0': 'Lomb-Scargle Periodogram Simple Max',
                'LS1': 'Lomb-Scargle Periodogram Tallest Peak',
                'LS2': 'Lomb-Scargle Periodogram HH Index',
                'CS0': 'Cubic Spline Num Roots',
                'CS1': 'Cubic Spline Integral'
                }

    @property
    def method_eval_dict(self):
        return {'PRNR': self.eval_PRNR,
                'NMC': self.eval_NMC,
                'MIMD': self.eval_MIMD,
                'MBPI': self.eval_MBPI,
                'WAVY': self.eval_WAVY,
                'FT0': self.eval_FT0,
                'FT1': self.eval_FT1,
                'FT2': self.eval_FT2,
                'LS0': self.eval_LS0,
                'LS1': self.eval_LS1,
                'LS2': self.eval_LS2,
                'CS0': self.eval_CS0,
                'CS1': self.eval_CS1
                }

    @property
    def method_g_dict(self):
        return {'PRNR': self.g_PRNR,
                'NMC': self.g_NMC,
                'MIMD': self.g_MIMD,
                'MBPI': self.g_MBPI,
                'WAVY': self.g_WAVY,
                'FT0': self.g_FT0,
                'FT1': self.g_FT1,
                'FT2': self.g_FT2,
                'LS0': self.g_LS0,
                'LS1': self.g_LS1,
                'LS2': self.g_LS2,
                'CS0': self.g_CS0,
                'CS1': self.g_CS1
                }

    @property
    def method_base_dict(self):
        return {'PRNR': b_PRNR,
                'NMC': b_NMC,
                'MIMD': b_MIMD,
                'MBPI': b_MBPI,
                'WAVY': b_WAVY,
                'FT0': b_FT0,
                'FT1': b_FT1,
                'FT2': b_FT2,
                'LS0': b_LS0,
                'LS1': b_LS1,
                'LS2': b_LS2,
                'CS0': b_CS0,
                'CS1': b_CS1
                }

    def calc_method_array(self, method, detrend_price=True):
        series = 'price'
        if detrend_price:
            series = 'detrended_price'
        if method in self.method_arrays:
            return
        b_func = self.method_base_dict[method]
        evaluations = np.ndarray(shape=self.n)
        if method in ['MBPI', 'DP']:
            evaluations = np.ndarray(shape=(self.n, 89))
        for i, ob in enumerate(self.obs_set):
            if method in ['PRNR', 'NMC', 'MIMD', 'MBPI']:
                evaluations[i] = b_func(ob.delta_price)
            else:
                evaluations[i] = b_func(ob.series_interface(series))
        self.method_arrays[method] = evaluations

    def evaluate_external_data(self, data, method):
        b_func = self.method_base_dict[method]
        evaluations = np.ndarray(shape=data.shape[0])
        if method in ['MBPI', 'DP']:
            evaluations = np.ndarray(shape=(data.shape[0], 89))
        for i in range(data.shape[0]):
            if method in ['PRNR', 'NMC', 'MIMD', 'MBPI']:
                evaluations[i] = b_func(
                    data[i, 1:] - data[i, :-1])  # delta price
            else:
                evaluations[i] = b_func(data[i, :])
        self.method_arrays[method] = evaluations

    def get_truth_array(self):
        truth = np.ndarray(shape=self.n, dtype='bool')
        for i, ob in enumerate(self.obs_set):
            truth[i] = ob.cycling_binary_interface(self.truth_criterion)
        return truth

    def setup_LL_func(self, method):
        def calc_LL(theta):
            ll = 0
            Lambda_array = Lambda(self.method_g_dict[method](theta))
            ll = log_l(self.truth, Lambda_array)
            return ll
        return calc_LL

    def MAGS(self, method, min_theta, max_theta,
             precision=2, resolution=10, window_cut_factor=4, objective='acc', verbose=1):
        '''
        Memoized Adaptive Grid Search - Simple optimization algorithm for
        poorly behaved functions. Hard coded here for convenience and 
        simplicity. No guarantees of optimality, but works well for this
        application.

        search domain: [min_theta, max_theta]

        Accuracy and time:
            - Proportional to resolution and precision
            - Inversely proportional to window_cut_factor
        '''
        timer = -1 * time.perf_counter()

        def acc_f(t):
            evaluations = self.method_eval_dict[method](t)
            return self.evaluate_accuracy(evaluations)

        if objective == 'LL':
            f = self.setup_LL_func(method)
        else:
            f = acc_f

        memo = {}
        l = min_theta
        r = max_theta
        w = r - l
        tol = math.pow(10, -1*precision)
        if verbose > 0:
            print(f'tol = {tol}')
        i = 0
        while True:
            memo_0 = len(memo)

            vals_this_iter = int(resolution+(i*resolution/2))
            w = r - l
            d = np.linspace(l, r, vals_this_iter).round(precision)
            if verbose > 0:
                print(f'iteration = {i}')
                print(
                    f'interval {vals_this_iter} values in = [{d[0]}, {d[-1]}]')
            if verbose > 1:
                print(f'w = {w:.4f}')
            for x in d:
                if memo.get(x) == None:
                    memo[x] = f(x)
            _max = np.max(list(memo.values()))
            argmax = [k for k, v in memo.items() if v == _max]
            argmax.sort()
            idx = int(len(argmax)/2)
            if verbose > 0:
                print(f'current max = {argmax[idx]} : {_max:.4f}')

            c = argmax[idx]
            l = max(c - w/window_cut_factor, l)
            r = min(c + w/window_cut_factor, r)
            if verbose > 0:
                print(f'cumulative function evaluations = {len(memo)}')
                print(' ')
            i += 1

            if len(memo) == memo_0:
                break
        if verbose > 0:
            print(f'function evaluations = {len(memo)}')
        timer += time.perf_counter()
        if verbose > 0:
            print(f'processing time =  {int(timer/60)}m  {timer%60:.1f}s\n')

        accuracy = _max
        theta = argmax[idx]

        sorted_memo = {}
        for key in sorted(memo.keys()):
            sorted_memo[key] = memo[key]
        return theta, accuracy, sorted_memo

    def evaluate_and_print_method(self, method, theta, theta_domain=[' ', ' ']):
        evaluations = self.method_eval_dict[method](theta)
        truth_dict = self.evaluate_truth(evaluations)
        heading = f'{self.region} - {self.method_name_dict[method]}'.upper()
        self.print_truth_table(truth_dict, heading, theta, theta_domain)
        return truth_dict

    def evaluate_theta(self, method, theta):
        evaluations = self.method_eval_dict[method](theta)
        accuracy = self.evaluate_accuracy(evaluations)
        return accuracy

    def eval_and_print_LL_func(self, method, theta, divide_by_N=False):
        ll_func = self.setup_LL_func(self.method_g_dict[method])
        ll_val = -1*ll_func(theta)
        if not divide_by_N:
            ll_val *= self.n
        print(f'{method} : theta = {theta}  ==>  LL = {ll_val:.2f}')

    def print_truth_table(self, truth_dict, heading=' ', theta=' ', theta_domain=[' ', ' '], subheading=True):
        col_widths = [19, 17, 17]
        perf = truth_dict['TT'] + truth_dict['FF']
        perf = perf / (perf + truth_dict['TF'] + truth_dict['FT']) * 100
        subhead = f'truth={self.truth_criterion},  domain=[{theta_domain[0]}, {theta_domain[1]}],  t*={str(theta)}'
        print('\n{:^{}}'.format(heading, np.sum(col_widths)))
        if subheading:
            print('{:^{}}'.format(subhead, np.sum(col_widths)))
        print('{:>{}}{:>{}}'.format(
            ' ', col_widths[0], 'METHOD EVALUATIONS', col_widths[1]+col_widths[2]))
        print('{:<{}}|{:>{}}{:>{}}'.format(
            'GROUND TRUTH', col_widths[0]-1, 'Cycling', col_widths[1], 'Not_Cycling', col_widths[2]))
        print('='*(np.sum(col_widths)))
        print('{:<{}}|{:>{}}{:>{}}'.format(
            'Cycling', col_widths[0]-1, truth_dict['TT'], col_widths[1], truth_dict['TF'], col_widths[2]))
        print('{:<{}}|{:>{}}{:>{}}'.format(
            'Not_Cycling', col_widths[0]-1, truth_dict['FT'], col_widths[1], truth_dict['FF'], col_widths[2]))
        print('-'*(np.sum(col_widths)))
        print(f'ACCURACY = {perf:.2f}%\n')

    def evaluate_truth(self, evaluations):
        '''
        first letter represents truth
        second letter represents evaluation
        '''
        truth_dict = defaultdict(int)
        for i in range(len(evaluations)):
            if self.truth[i] == True and evaluations[i] == True:
                truth_dict['TT'] += 1
            elif self.truth[i] == False and evaluations[i] == False:
                truth_dict['FF'] += 1
            elif self.truth[i] == False and evaluations[i] == True:
                truth_dict['FT'] += 1  # Type 1 error (false positive)
            elif self.truth[i] == True and evaluations[i] == False:
                truth_dict['TF'] += 1  # Type 2 error (false negative)
        return truth_dict

    def g_PRNR(self, theta):
        if 'PRNR' not in self.method_arrays:
            self.calc_method_array('PRNR')
        return self.method_arrays['PRNR'] - theta

    def g_NMC(self, theta):
        if 'NMC' not in self.method_arrays:
            self.calc_method_array('NMC')
        return self.method_arrays['NMC'] - theta

    def g_MIMD(self, theta):
        if 'MIMD' not in self.method_arrays:
            self.calc_method_array('MIMD')
        return self.method_arrays['MIMD'] - theta

    def g_MBPI(self, theta1):
        if 'MBPI' not in self.method_arrays:
            self.calc_method_array('MBPI')
        cycle_count_array = np.count_nonzero(
            self.method_arrays['MBPI'] > theta1, axis=1)
        return cycle_count_array

    def g_WAVY(self, theta):
        if 'WAVY' not in self.method_arrays:
            self.calc_method_array('WAVY')
        return self.method_arrays['WAVY'] - theta

    def g_FT0(self, theta):
        if 'FT0' not in self.method_arrays:
            self.calc_method_array('FT0')
        return self.method_arrays['FT0'] - theta

    def g_FT1(self, theta):
        if 'FT1' not in self.method_arrays:
            self.calc_method_array('FT1')
        return self.method_arrays['FT1'] - theta

    def g_FT2(self, theta):
        if 'FT2' not in self.method_arrays:
            self.calc_method_array('FT2')
        return self.method_arrays['FT2'] - theta

    def g_LS0(self, theta):
        if 'LS0' not in self.method_arrays:
            self.calc_method_array('LS0')
        return self.method_arrays['LS0'] - theta

    def g_LS1(self, theta):
        if 'LS1' not in self.method_arrays:
            self.calc_method_array('LS1')
        return self.method_arrays['LS1'] - theta

    def g_LS2(self, theta):
        if 'LS2' not in self.method_arrays:
            self.calc_method_array('LS2')
        return self.method_arrays['LS2'] - theta

    def g_CS0(self, theta):
        if 'CS0' not in self.method_arrays:
            self.calc_method_array('CS0')
        return self.method_arrays['CS0'] - theta

    def g_CS1(self, theta):
        if 'CS1' not in self.method_arrays:
            self.calc_method_array('CS1')
        return self.method_arrays['CS1'] - theta

    def eval_PRNR(self, theta):
        return self.g_PRNR(theta) < 0

    def eval_NMC(self, theta):
        return self.g_NMC(theta) < 0

    def eval_MIMD(self, theta):
        return self.g_MIMD(theta) > 0

    def eval_WAVY(self, theta):
        return self.g_WAVY(theta) > 0

    def eval_FT0(self, theta):
        return self.g_FT0(theta) > 0

    def eval_FT1(self, theta):
        return self.g_FT1(theta) > 0

    def eval_FT2(self, theta):
        return self.g_FT2(theta) > 0

    def eval_LS0(self, theta):
        return self.g_LS0(theta) > 0

    def eval_LS1(self, theta):
        return self.g_LS1(theta) > 0

    def eval_LS2(self, theta):
        return self.g_LS2(theta) > 0

    def eval_CS0(self, theta):
        return self.g_CS0(theta) > 0

    def eval_CS1(self, theta):
        return self.g_CS1(theta) > 0

    def eval_MBPI(self, theta):
        '''
        if theta is list it will interpret it to be
        [theta1, theta2] else
        search entire space of theta2 since it is
        integer value, and return the highest accuracy
        result, in this way theta2 becomes implicit
        '''
        if type(theta) != list:
            theta1 = theta
            cycle_count_array = self.g_MBPI(theta1)
            return_theta2 = 0
            max_acc = 0
            for theta2 in range(46):
                evals = cycle_count_array >= theta2
                acc = self.evaluate_accuracy(evals)
                if acc > max_acc:
                    max_acc = acc
                    return_theta2 = theta2
        else:
            theta1 = theta[0]
            return_theta2 = theta[1]
            cycle_count_array = self.g_MBPI(theta1)
        return cycle_count_array >= return_theta2

    def retrieve_MBPI_theta2(self, theta1):
        cycle_count_array = self.g_MBPI(theta1)
        return_theta2 = 0
        max_acc = 0
        for theta2 in range(46):
            evals = cycle_count_array >= theta2
            acc = self.evaluate_accuracy(evals)
            if acc > max_acc:
                max_acc = acc
                return_theta2 = theta2
        return return_theta2

    def evaluate_accuracy(self, evaluations):
        n_correct = np.count_nonzero(self.truth == evaluations)
        percent_correct = n_correct/len(self.truth) * 100
        return percent_correct
