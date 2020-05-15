from __future__ import absolute_import, division, print_function
import numpy as np
import glob
import os
import pickle
import bz2


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
