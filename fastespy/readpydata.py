from __future__ import absolute_import, division, print_function
import numpy as np
import glob
import os

def readgraphpy(directory, split = '-', checkroot = True, inputid = 'in', prefix = ''):

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