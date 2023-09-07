'''
Author: Timothy Holt - tabholt@gmail.com
Aug 2023

Utility class for measuring and managing time intervals in Python applications.

The Timer class provides a simple and flexible way to track and record time
intervals within a program. It can be used for benchmarking, profiling, or
simply monitoring the execution time of specific code blocks.

Attributes:
    - name (str): A name or description for the Timer (optional).
    - string_reset (bool): If True, the timer resets after getting the time.
    - basic_string (bool): If True, the string representation provides a basic
      time measurement (e.g., '5.2s'). Otherwise, it includes a label (e.g.,
      'time = 5.2 s').

Methods:
    - time: Returns the current elapsed time since the timer's last reset.
    - get(name): Returns the recorded time for a specific checkpoint.
    - reset: Resets the timer to its initial state.
    - checkpoint(name): Records the current time and associates it with a
      checkpoint name.
    - save_and_reset(name): Records the current time, associates it with a
      checkpoint name, and resets the timer.
    - print_all_timers: Prints all recorded times with optional human-readable
      formatting.
    - seconds_to_hms_string(seconds): Converts seconds to a formatted
      hours:minutes:seconds string.
    - print_all_timers_seconds: Prints all recorded times in seconds.
    - write_all_timers_to_file(filename): Writes all recorded times to a file.

Example Usage:
    - Create a Timer object, record times using checkpoints, and print or save
      the recorded times.

Note:
    - The Timer class relies on the time module for time measurements.

'''

# The Timer class provides a convenient way to measure time intervals and
# track execution times within Python applications. It is useful for various
# purposes, including profiling code blocks, benchmarking, or monitoring
# performance.

# Example Usage:
# timer = Timer("Example Timer")
# timer.checkpoint("Start")
# # ... Perform some operations ...
# timer.checkpoint("Middle")
# # ... Perform more operations ...
# timer.checkpoint("End")
# timer.print_all_timers()
# timer.write_all_timers_to_file("timers.txt")


import time

class Timer(object):
    def __init__(self, name='', string_reset=True, basic_string=False):
        self.name = name
        self.string_reset = string_reset
        self.basic_string = basic_string
        self.timer: float
        self.timers = {}
        self.reset()
        self.hms_threshold = 240

    @property
    def time(self):
        return self.timer+time.perf_counter()

    def get(self, name):
        return self.timers[name]

    def reset(self):
        self.timer = -1 * time.perf_counter()

    def __str__(self) -> str:
        t = self.time
        if self.string_reset:
            self.reset()
        if self.basic_string:
            return f'{t:.1f}s'
        return f'time = {t:.1f} s  '

    def checkpoint(self, name):
        t = self.time
        self.timers[name] = t
        return t

    def save_and_reset(self, name):
        t = self.time
        self.timers[name] = t
        self.reset()
        return t

    def print_all_timers(self):
        hms_version = False
        for v in self.timers.values():
            if v > self.hms_threshold:
                hms_version = True
        if not hms_version:
            self.print_all_timers_seconds()
            return
        print(f'\n{self.name.upper()} TIMERS:')
        total = 0
        longest_title = 0
        for k in self.timers.keys():
            if len(k) > longest_title:
                longest_title = len(k)
        for k, v in self.timers.items():
            hms_string = self.seconds_to_hms_string(v)
            print(f'{k.ljust(longest_title+1)}={hms_string}')
            total += v
        print('-'*(longest_title+2+len(hms_string)))
        tot = 'total_time'
        hms_string = self.seconds_to_hms_string(total)
        print(f'{tot.ljust(longest_title+1)}={hms_string}\n')

    def seconds_to_hms_string(self, seconds):
        t_h = int(seconds/3600)
        t_m = int((seconds - t_h*3600)/60)
        t_s = seconds - t_h*3600 - t_m*60
        h_str = ' '
        m_str = ' '
        s_str = f'{t_s:.1f}s'
        if t_h:
            h_str += f'{t_h}h'
        if t_m:
            m_str += f'{t_m}m'
        return f'{h_str:>3}{m_str:>4}{s_str:>6}'

    def print_all_timers_seconds(self):
        print(f'\n{self.name.upper()} TIMERS:')
        longest_title = 0
        for k in self.timers.keys():
            if len(k) > longest_title:
                longest_title = len(k)
        total = 0
        for k, v in self.timers.items():
            print(f'{k.ljust(longest_title+1)}={v:7.2f} s')
            total += v
        print('-'*(longest_title+11))
        tot = 'total_time'
        print(f'{tot.ljust(longest_title+1)}={total:7.2f} s\n')

    def write_all_timers_to_file(self, filename):
        text = ''
        for k, v in self.timers.items():
            text += f'{k},{v:.2f},'
        text = text[:-1] + '\n'
        with open(filename, 'a') as f:
            f.write(text)
