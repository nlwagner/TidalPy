import time

import numpy as np

from .conversions import convert_to_hms

def progress_bar(loop_limits: tuple, verbose: bool = True):

    def outer_wrapper(function):

        clock_old = time.time()
        prev_delta = 1.
        prev_prev_delta = 1.
        prev_perc = 0.

        def inner_wrapper(input_time, *args, **kwargs):

            diff_loop = kwargs.get('diff_loop', False)
            now = time.time()
            # Print Progress Bar
            if verbose and diff_loop:
                if now - inner_wrapper.clock_old > 1.5:
                    percent_done = (input_time - loop_limits[0]) / (loop_limits[-1] - loop_limits[0])
                    percent_done = np.nan_to_num(percent_done)
                    delta = (percent_done - inner_wrapper.prev_perc) / (now - inner_wrapper.clock_old)
                    time_left = 3. * (1. - percent_done) / (delta + inner_wrapper.prev_delta + inner_wrapper.prev_prev_delta)
                    if time_left > 99999999.0:
                        time_left = 99999999.0
                    print('\rPercent Done: {:0>5.2f}%. Approx. Time Remaining: {:0>2} days, {:0>2.0f}:{:0>2.0f}::{:0>4.1f}'.format(
                            100. * percent_done, *convert_to_hms(time_left)), flush=True, end='')
                    inner_wrapper.prev_perc = percent_done
                    inner_wrapper.clock_old = now
                    inner_wrapper.prev_prev_delta = max(inner_wrapper.prev_delta, 1.e-5)
                    inner_wrapper.prev_delta = max(delta, 1.e-5)

            return function(input_time, *args, **kwargs)

        inner_wrapper.clock_old = clock_old
        inner_wrapper.prev_delta = prev_delta
        inner_wrapper.prev_perc = prev_perc
        inner_wrapper.prev_prev_delta = prev_prev_delta

        return inner_wrapper

    return outer_wrapper