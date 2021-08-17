from __future__ import absolute_import, division, print_function
from scipy import signal
from scipy.special import gammaln
import numpy as np
import logging
import sys


def init_logging(level, color=False):
    """
    Setup logger.

    Parameters
    ----------
    level:        string, level of logging: DEBUG,INFO,WARNING,ERROR. (default: INFO).

    kwargs
    ------
    color:        bool, if true, enable colored output for bash output

    Notes
    -----
    for color see
        stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
        https://wiki.archlinux.org/index.php/Color_Bash_Prompt
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if level.upper() == 'INFO':
        level = logging.INFO
    elif level.upper() == 'DEBUG':
        level = logging.DEBUG
    elif level.upper() == 'WARNING':
        level = logging.WARNING
    elif level.upper() == 'ERROR':
        level = logging.ERROR


    if color:
        logging.basicConfig(level=level,stream = sys.stderr, format='\033[0;36m%(filename)10s:\033[0;35m%(lineno)4s\033[0;0m --- %(levelname)7s: %(message)s')
        logging.addLevelName( logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
        logging.addLevelName( logging.INFO, "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
        logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    else:
        logging.basicConfig(level=level,stream = sys.stderr, format='%(filename)10s:%(lineno)4s --- %(levelname)7s: %(message)s')

    return

def opt_bins(data, maxM=100):
    """
    Python version of the 'optBINS' algorithm by Knuth et al. (2006) - finds
    the optimal number of bins for a one-dimensional data set using the
    posterior probability for the number of bins. WARNING sometimes doesn't
    seem to produce a high enough number by some way...

    Adopted from https://github.com/samconnolly/DELightcurveSimulation/blob/master/DELCgen.py
    Ref: https://arxiv.org/pdf/physics/0605197.pdf

    :param data: array-like
        The data set to be binned
    :param maxM: int, optional
        The maximum number of bins to consider

    :return: int
        The optimal number of bins
    """

    # loop through the different numbers of bins
    # and compute the posterior probability for each.

    logp = np.zeros(maxM)
    for M in range(1, maxM+1):
        n = np.histogram(data, bins=M)[0]  # Bin the data (equal width bins)
        # calculate posterior probability
        part1 = len(data) * np.log(M) + gammaln(M/2.0)
        part2 = - M * gammaln(0.5) - gammaln(len(data) + M/2.0)
        part3 = np.sum(gammaln(n+0.5))
        logp[M-1] = part1 + part2 + part3  # add to array of posteriors
        maximum = np.argmax(logp) + 1  # find bin number of maximum probability
    return maximum + 10



def periodogram(t, v, fsample, timeslice=None, window='hanning'):
    """
    Calculate Periodogram for a list of measurements

    :param t: array-like
        List of array of time measurements
    :param v:
        List of array of voltage measurements
    :param fsample: float
        Sampling frequency in Hz
    :param timeslice: float
        time slice in units 1/fsample. Arrays will be split up
        into slices of this length and periodogram will be averaged.
        If None, full time range is used.
    :param window: str
        Window function for periodogram. Default: 'hanning'
    :return:
    """
    total_time = [len(t[i]) / fsample for i in range(len(t))]
    if timeslice is None:
        timeslice = total_time

    npts = [int(len(t[i]) / (total_time[i] / timeslice)) for i in range(len(t))]

    maxidx = [int((total_time[i] / timeslice)) * int(len(t[i]) / (total_time[i] / timeslice))
              for i in range(len(t))]
    vreshape = np.vstack([v[i][:maxidx[i]].reshape(int((total_time[i] / timeslice)), npts[i])
                         for i in range(len(t))])

    fs, ps = signal.periodogram(vreshape, fs=fsample, axis=-1, window=window)
    return fs, ps

def hist_norm(v, bins = None):
    """
    Compute histogram of voltages as well as mean and variance

    :param v: array-like
        voltage measurements
    :param bins: array-like or int
        bins for histogram. If None, compute with opt_bins function
    :return: bin values, bin centers, bins, mean, variance of v
    """
    if bins is None:
        bins = opt_bins(v)

    n,bins = np.histogram(v, bins = bins, density = True)
    vc = 0.5 * (bins[1:] + bins[:-1])
    return n,vc,bins,v.mean(), v.var()

def derivative(v):
    """
    Compute derivative of finite series

    :param v: array-like
        time series
    :return: array
        finite derivative
    """
    dv = 0.5 * (v[2:] - v[:-2])
    dv = np.insert(dv, 0, v[1] - v[0])
    dv = np.append(dv, v[-1] - v[-2])
    return dv

def filter(x, fSample, fmax = 1.e6, norder = 3):

    # calculate the Nyquist frequency
    fNyq = x.size / 2. /(x.size - 1) * fSample
    if norder > 0:
        b, a = signal.butter(norder, fmax / fNyq)
        xf = signal.filtfilt(b, a, x)
    else:
        xf = x
    return xf

def derivative_filtered(v,fSample, fmax = 1.e6, norder = 3):
    """
    Calculate the derivative and apply a filter

    :param t: array-like
        time values
    :param v: array-like
        voltage values
    :param fSample: float
        sampling frequency
    :param fmax:
        maximum frequency above which filter is applied, see scipy.signal.butter function
    :param norder:
        filter order
    :return:
        derivative values and filtered derivative values
    """
    # calculate derivative
    dv = derivative(v) * fSample
    # if filter order larger than 0, apply filter above fmax frequency
    dvf = filter(dv, fSample, fmax=fmax, norder=norder)
    return dv, dvf

def build_trigger_windows(t,v,fSample, thr = -50., thr_v = -0.001,
    tstepup=10., tsteplo=5., fmax = 1.e6, norder = 3):
    """
    Build trigger windows from a continuous time line

    :param t: array-like
        time values in seconds
    :param v: array-like
        voltage values in volts
    :param fSample: float
        sampling frequency
    :param thr: float
        threshold value for voltage derivative for trigger.
        In units mV / micro s, default is -50.
    :param thr_v: float
        threshold value for voltage amplitude
        In units V, default is -0.001 V = -1. mV
    :param tstepup: float
        time window after trigger, in micro s, default is 50.
        No additional trigger within this time will be considered.
    :param tsteplo:
    :param tsteplo:
        time window before trigger, in micro s, default is 10.
        No additional trigger within this time will be considered.
    :param fmax:
        maximum frequency for filtering voltage in Hz. Default is 1 MHz
    :param norder:
        filter order, default is 3
    :return:
        three lists with trigger times and times and voltage values of trigger windows.
    """

    dv, dvf = derivative_filtered(v, fSample, fmax=fmax, norder=norder)

    mtrig = np.where((dvf * 1e3 / 1e6 < thr) & (v < thr_v))[0] # convert dv to mV / micro s

    #idxs = np.where(np.diff(mtrig) > 1)[0]  # find indeces that are non-consecutive
    idxs = np.where(np.diff(mtrig) > int(tstepup * 1e-6 * fSample))[0]  # find indeces that are larger than window

    t0 = []
    t_trig = []
    v_trig = []

    if not len(mtrig):
        raise ValueError("No triggers found, decrease threshold (current: dvdt = {0:.2f} mV / mu s"\
                         ", v = {1:.2e} V)".format(thr, thr_v))
    print ("{0:n} trigger(s) found!".format(len(idxs)+1))

    # build first trigger window
    idlo = mtrig[0] - int(tsteplo * 1e-6 * fSample)
    idup = mtrig[0] + int(tstepup * 1e-6 * fSample)
    ps = slice(idlo, idup)
    t_trig.append(t[ps])
    v_trig.append(v[ps])
    t0.append(t[mtrig[0]])

    # build remaining trigger windows
    for i, ii in enumerate(idxs):
        iitrig = ii+1
        idlo = mtrig[iitrig] - int(tsteplo * 1e-6 * fSample)
        idup = mtrig[iitrig] + int(tstepup * 1e-6 * fSample)

        if idlo < 0:
            idlo = 0
        if idup > t.size:
            idup = t.size

        ps = slice(idlo, idup)
        t_trig.append(t[ps])
        v_trig.append(v[ps])
        t0.append(t[mtrig[iitrig]])

    return t0, t_trig, v_trig
