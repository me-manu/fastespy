from __future__ import absolute_import, division, print_function
import ROOT as root
import glob
import os
import numpy as np
import time

def readgraph(directory, split = '-', overwrite = False, inputid = 'in', prefix = ''):

    """
    Read data from a ROOT graph and save as numpy npz file.

    :param directory: str
    Directory containing the root files.

    :return:
    """
    files = glob.glob(os.path.join(directory, "{0:s}*.root".format(prefix)))
    files = sorted(files, key=lambda f: int(os.path.basename(f).split(".")[0].split(split)[-1]))

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
