from __future__ import absolute_import, division, print_function
import ROOT as root
import glob
import os
import numpy as np
import time
import logging


def readgraph(directory, split = '-', overwrite = False, inputid = 'in', prefix = ''):
    """
    Read data from a ROOT graph and save as numpy npz file.

    :param directory: str
    Directory containing the root files.

    :return:
    """
    files = glob.glob(os.path.join(directory, "{0:s}*.root".format(prefix)))
    try:
        files = sorted(files, key=lambda f: int(os.path.basename(f).split(".")[0].split(split)[-1]))
    except ValueError as e:
        print("Warning: could not sort root files by name: {0}".format(e))

    npzfiles = [f.replace('.root', '.npz') for f in files]

    t = []
    v = []
    tin = []
    vin = []

    for i, rootfile in enumerate(files):
        if os.path.isfile(npzfiles[i]) and not overwrite:
            print("Reading file {0:n} of {1:n}: {2:s}".format(i + 1, len(files), npzfiles[i]))
            r = np.load(npzfiles[i])
            x = r['x']
            y = r['y']

        else:
            t1 = time.time()
            print("Reading file {0:n} of {1:n}: {2:s}".format(i + 1, len(files), rootfile))
            f = root.TFile.Open(rootfile)
            hh = root.TGraph()
            if inputid in rootfile:
                root.gDirectory.GetObject('Data of channel ChB', hh)
                x = np.array([hh.GetX()[i] for i in range(len(hh.GetX()))])
                y = np.array([hh.GetY()[i] for i in range(len(hh.GetY()))])
            else:
                root.gDirectory.GetObject('Data of channel ChA', hh)
                x = np.array([hh.GetX()[i] for i in range(len(hh.GetX()))])
                y = np.array([hh.GetY()[i] for i in range(len(hh.GetY()))])
            f.Close()
            print("Reading root file took {0:.2f} minutes".format((time.time() - t1) / 60.))

            np.savez(npzfiles[i], x=x, y=y)

        if inputid in rootfile:
            tin.append(x)
            vin.append(y)
        else:
            t.append(x)
            v.append(y)

    return t, v, tin, vin


def root2py_fit_results_axel(path,
                             tree_name="TES_fits",
                             features=("ampli", "peak", "rise", "decay", "const", "chi2")):
    """
    Read in Axel's fit results obtained with ROOT
    and save as npy file

    Parameters
    ----------
    path: str
        path containing the root files

    tree_name: str
        Name of root tree where results are stored

    features: list of strings
        feature names that are extracted from root tree
    """
    data_files = glob.glob(os.path.join(path, '*.root'))
    if not len(data_files):
        raise ValueError("No files found!")

    for i, d in enumerate(data_files):

        logging.info("Reading file {0:s}, assigning event id {1:n}".format(d, i))
        r = root.TFile(d)
        t = root.TTree()
        r.GetObject(tree_name, t)
        n_events = t.GetEntries()

        result = dict()
        for k in features:
            t.Draw(k)
            vals = t.GetVal(0)
            result[k] = np.zeros(n_events)
            for j in range(n_events):
                result[k][j] = vals[j]
        result['type'] = np.full(n_events, i)

        del r, t
        filename = os.path.join(os.path.dirname(d), os.path.basename(d).replace('root','npy'))
        logging.info(f"Saving fit results to {filename}")
        np.save(filename, result)
