"""
Some misc. utilities.
"""

import numpy as np

DEFAULT_N_JOBS = 8


def correct_lag(lag, neurons, motion):
    """
    Shifts timeseries to account for an optimal lag between neurons and motion.
    Positive lag indicates that motion preceds neurons.
    """

    if lag > 0:
        neurons = neurons[:-lag, :]
        if motion is not None:
            motion = motion[lag:, :]
    elif lag < 0:
        neurons = neurons[-lag:, :]
        if motion is not None:
            motion = motion[:lag, :]

    return neurons, motion


def get_nneur(n_neurons, n_sample):
    """
    Raise a warning if n_sample is greater than the
    next power of two largest than n_neurons.
    Otherwise return min(n_sample, n_neurons).
    This is helpful when we are sampling neurons in powers
    of two, but also want to sample the maximum neuron number.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in dataset
    n_sample : int
        Number of desired neurons to sample,
        assumed to be a power of 2
    """

    power_2 = int(np.log2(n_sample))

    if n_sample == 0 or (n_sample > n_neurons and n_neurons > (2 ** (power_2 - 1))):
        n_sample = n_neurons
        print("nneur set to", n_sample)

    if n_neurons < n_sample:
        log = "nneur " + str(n_sample) + " > total # neurons " + str(n_neurons)
        raise ValueError(log)

    return n_sample
