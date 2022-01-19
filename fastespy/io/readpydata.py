from __future__ import absolute_import, division, print_function
import numpy as np
import glob
import os
import pickle
import bz2
import logging
import copy
import tqdm
from ..timeline.models import TimeLine


def read_graph_py(directory, split = '-', checkroot = True, inputid = 'in', prefix = ''):
    """
    Read data from a ROOT graph and save as numpy npz file.

    :param directory: str
    Directory containing the root files.

    :return:
    """
    if checkroot:
        files = glob.glob(os.path.join(directory, "{0:s}*.root".format(prefix)))
        files = sorted(files, key=lambda f: int(os.path.basename(f).split(".")[0].split(split)[-1]))
        npzfiles = [f.replace('.root', '.npz') for f in files]
        if not len(files) == len(npzfiles):
            raise ValueError("Not the same number of root and npz files, convert data first!")
    else:
        npzfiles =glob.glob(os.path.join(directory, "{0:s}*.npz".format(prefix)))
        npzfiles = sorted(npzfiles, key=lambda f: int(os.path.basename(f).split(".")[0].split(split)[-1]))

    t = []
    v = []
    tin = []
    vin = []

    for i, npz in enumerate(npzfiles):
        print("Reading file {0:n} of {1:n}: {2:s}".format(i + 1, len(npzfiles), npzfiles[i]))
        r = np.load(npz)
        x = r['x']
        y = r['y']

        if inputid in npz:
            tin.append(x)
            vin.append(y)
        else:
            t.append(x)
            v.append(y)

    return t, v, tin, vin


def read_converted_pickle(picklefile, channel=0):
    """
    Function to read in converted root raw file
    after running script fastespy/fastespy/scripts/convert_root_file_to_python.py

    :param picklefile: str
        path to pickle file
    :param channel: int
        Channel id, either 0 (channel A, default), or 1 (channel B).
    :return: list
        unplickled list of dictionaries of triggers
    """

    with bz2.BZ2File(picklefile, "rb") as f:
        r = pickle.load(f)[channel]

    # r is now a list of dictionaries, one dict per trigger
    # convert the data to numpy arrays
    # and add a time key
    for i in range(len(r)):
        r[i]['data'] = np.array(r[i]['data'])
        r[i]['time'] = r[i]['timeStamp'] + \
                       np.arange(0., r[i]['data'].size / r[i]['samplingFreq'], 1. / r[i]['samplingFreq'])

    return r


def read_fit_results_rikhav(path,
                            feature_names=('pulse integral fit',
                                'amplitude', 'rise time', 'decay time', 'chi2 reduced')
                            ):
    """
    Read in the fit results from Rikhav's fit
    and concatenate them to one file.


    Parameters
    ----------
    path: str
        Path to Rikhav's data files

    feature_names: list of str
        feature names as given in the fit result file

    Returns
    -------
    Dict with fit results and list with feature names
    """

    data_files = sorted(glob.glob(os.path.join(path, '*.npy')))
    if not len(data_files):
        raise ValueError("No data files found!")

    result = dict()
    result['type'] = []
    for k in feature_names:
        result[k.replace(' ', '_')] = []
    result['chi2'] = []
    result['t0'] = []
    result['tmax'] = []
    result['integral'] = []
    tl = TimeLine(numcomp=1, function='expflare')

    for i, df in enumerate(data_files):
        logging.info(f"Reading file {df:s}, assigned type: {i}")
        x = np.load(df, allow_pickle=True).tolist()
        for xi in x.values():
            for k in feature_names:
                result[k.replace(' ', '_')].append(xi[k])
            result['type'].append(i)
            result['chi2'].append(xi['chi2 reduced'] * (xi['data'].size - 4))

            result['t0'].append(xi['time'][0])
            result['tmax'].append(xi['time'][-1])

            result['integral'].append(tl.integral(0., 100.,
                                                  tstep=1000,
                                                  t0_000=10.,
                                                  tr_000=result['rise_time'][-1] * 1e6,  # s to micro s
                                                  td_000=result['decay_time'][-1] * 1e6,  # s to micro s
                                                  A_000=-result['amplitude'][-1],
                                                  c=0.)[0])  # integral in (micro s) * V
            if not np.isfinite(result['integral'][-1]):
                result['integral'][-1] = 1e20

        del x

    for k, v in result.items():
        result[k] = np.array(result[k])
    return result


def read_fit_results_manuel(path):
    """
    Read in the fit results from Manuel's fit
    and concatenate them


    Parameters
    ----------
    path: str
        Path to Manuel's fit result files

    Returns
    -------
    Dict with fit results and list with feature names
    """
    data_files = glob.glob(os.path.join(path, "*.npy"))
    if not len(data_files):
        raise ValueError("No data files found!")

    for i, result_file in enumerate(data_files):
        r = np.load(result_file, allow_pickle=True).flat[0]
        logging.info("Reading file {0:s} of data type {1:n}".format(result_file, r['type'][0]))
        if not i:
            result = r
            result['chi2'] = r['chi2_dof'] * r['dof']
        else:
            for k, v in result.items():
                if k == 'chi2':
                    result['chi2'] = np.append(v, r['chi2_dof'] * r['dof'])
                else:
                    result[k] = np.append(v, r[k])

    return result


def read_fit_results_axel(path):
    """
    Read in the fit results from Axel's fit
    converted from a root file and concatenate them


    Parameters
    ----------
    path: str
        Path to Manuel's fit result files

    Returns
    -------
    Dict with fit results and list with feature names
    """

    data_files = glob.glob(os.path.join(path, "*.npy"))

    tl = TimeLine(numcomp=1, function='expflare')
    if not len(data_files):
        raise ValueError("No data files found!")

    for i, result_file in enumerate(data_files):
        r = np.load(result_file, allow_pickle=True).flat[0]
        logging.info("Reading file {0:s} of data type {1:n}".format(result_file, r['type'][0]))

        r['integral'] = np.zeros_like(r['rise'])
        # add the integral
        for j in range(r['rise'].size):
            r['integral'][j] = tl.integral(0., 100.,
                                           tstep=1000,
                                           t0_000=r['peak'][j] * 1e6,  # s to micro s
                                           tr_000=r['rise'][j] * 1e6,  # s to micro s
                                           td_000=r['decay'][j] * 1e6,  # s to micro s
                                           A_000=-r['ampli'][j],
                                           c=0.)  # integral in (micro s) * V
            if not np.isfinite(r['integral'][j]):
                r['integral'][j] = 1e20
        if not i:
            result = r
        else:
            for k, v in result.items():
                result[k] = np.append(v, r[k])

    return result


def convert_data_to_ML_format(result, features, bkg_type, signal_type):
    """
    Convert fit result dictionary to ML data format

    Parameters
    ----------
    result: dict
        Dictionary with fit results, must contain a 'type' key
    features: list of strings
        list with feature names that are extracted for ML
    bkg_type: int
        Type value for the background
    signal_type: int
        Type value for the signal

    Returns
    -------
    Data vector X of shape n_samples x n_features and class array of shape n_samples,
    signal events are class 1, background is class 0
    """

    X_data = np.zeros((result['type'].size, len(features)))
    y_data = np.array(result['type']).astype(np.int)

    for i, k in enumerate(features):
        X_data[:, i] = copy.deepcopy(result[k])

    m = (result['type'] == bkg_type) | (result['type'] == signal_type)
    X = X_data[m]

    y = np.zeros(y_data[m].size, dtype=np.int)
    y[y_data[m] == signal_type] = 1

    return X, y


def load_data_rikhav(files, feature_names, light_cleaning_cuts={}):
    """
    Load data from Rikhav's pulse fitting

    Parameters
    ----------
    files: list
        list of file names

    feature_names: list
        list of feature names to be used

    light_cleaning_cuts: dict
        dictionary with feature names as key and cleaning options to be used

    Returns
    -------
    Tuple containing:
        - Dictionary with fitting data
        - Dictionary for raw data
        - Float with total observation time
    """

    result = {'type': []}
    t_tot_hrs = 0.
    id_rejected = []
    data = {"time": [], "data": []}
    # loop through files
    logging.info("Reading data")

    for f in tqdm.tqdm(files):
        x = np.load(f, allow_pickle=True).tolist()

        # for each file: calculate observation time
        t_start = 1e10
        t_stop = 0.

        if 'light' in str(f):
            id_rejected.append([])

        # loop through triggers
        for i in range(1, len(x.keys()) + 1):
            # light sample cleaning
            if 'light' in str(f):
                m = True
                for c, v in light_cleaning_cuts.items():
                    # print(i, v, {c.split()[0]: x[i][c]})
                    m &= eval(v, {c.split()[0]: x[i][c]})

                if not m:
                    id_rejected[-1].append(i)
                    continue

            for name in feature_names:
                if not name in result.keys():
                    result[name] = []

                result[name].append(x[i][name])

            # save raw data
            data['time'].append(x[i]['time'])
            data['data'].append(x[i]['data'])

            if 'intrinsic' in str(f) or 'extrinsic' in str(f):
                if x[i]['end time in hrs'] > t_stop:
                    t_stop = x[i]['end time in hrs']
                if x[i]['start time in hrs'] < t_start:
                    t_start = x[i]['start time in hrs']
                result['type'].append(0)

            if 'light' in str(f):
                result['type'].append(1)

        if 'intrinsic' in str(f):
            t_tot_hrs += t_stop - t_start  # only add for dark count rate
    for rej in id_rejected:
        logging.info("Rejected {0:n} triggers in light file".format(len(rej)))
    for k in ['time', 'data']:
        data[k] = np.array(data[k])
    # convert into into numpy arrays
    for k, v in result.items():
        if k == 'type':
            dtype = np.bool
        else:
            dtype = np.float32
        result[k] = np.array(v, dtype=dtype)

    logging.info("In total, there are {0:n} light events and {1:n} background events"
                 " for an observation time of {2:.2f} hours".format(result['type'].sum(),
                                                                    np.invert(result['type']).sum(),
                                                                    t_tot_hrs
                                                                    ))
    return result, data, t_tot_hrs
