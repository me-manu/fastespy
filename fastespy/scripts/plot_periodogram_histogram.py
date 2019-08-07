from __future__ import absolute_import, division, print_function
import argparse
import os
import glob
import numpy as np
from fastespy.readpydata import readgraphpy
from fastespy.analysis import periodogram, hist_norm
from scipy.stats import norm
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    usage = "usage: %(prog)s -d directory [-s suffix]"
    description = "Compute and plot periodogram and histograms of time line"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-d', '--directory', required=True, help='Directory with npz data files')
    parser.add_argument('-s', '--suffix', help='suffix for data files', default='tes2')
    parser.add_argument('-m', '--maskvalue', help='voltage values above which values are used',
                        type=float, default=0.)
    args = parser.parse_args()

    if not len(glob.glob(os.path.join(args.directory, "{0:s}*.root".format(args.suffix)))) == len(
            glob.glob(os.path.join(args.directory, "{0:s}*.npz".format(args.suffix)))):
        raise IOError("Error: convert files first!")

    t, v, tin, vin = readgraphpy(args.directory, prefix=args.suffix)

    # set the mask for the voltage
    for i, vv in enumerate(vin):
        plt.hist(vv, bins=100, density=True, alpha=0.5)
        plt.xlabel('Voltage (V)')
        plt.savefig(os.path.join(args.directory,'hist_voltages.png'))
        plt.close("all")

    m = []
    if len(vin):
        for i, vv in enumerate(vin):
            m.append(vv > args.maskvalue)
    else:
        for i, vv in enumerate(v):
            m.append(np.ones(vv.size, dtype=np.bool))

    # plot the time line
    plt.figure(figsize=(12, 4))
    plt.plot(t[0][m[0]], v[0][m[0]], marker='.', ms=1., alpha=0.5, ls='none')

    if len(vin):
        plt.plot(tin[0], vin[0])
    plt.savefig(os.path.join(args.directory,'time_line.png'))
    plt.close("all")

    fs, ps = periodogram([t[i][m[i]] for i in range(len(t))],
                         [v[i][m[i]] for i in range(len(t))],
                         fsample=20e6, timeslice=0.01,
                         window='hanning')
    plt.loglog(fs, ps.mean(axis=0))
    plt.ylabel("Power")
    plt.xlabel("Frequency (Hz)")
    plt.savefig(os.path.join(args.directory,'periodogram.png'))
    plt.close("all")

    # Plot the histogram of concatenated data
    vtot = np.concatenate([v[i][m[i]] for i in range(len(t))])
    ttot = np.concatenate([t[i][m[i]] for i in range(len(t))])

    plt.figure(figsize=(8, 6))

    ax = plt.subplot(111)
    n, vc, bins, mean, var = hist_norm(vtot, bins=100)

    plt.semilogy(vc, n)
    plt.semilogy(vc, norm.pdf(vc, loc=mean, scale=np.sqrt(var)), ls='--')
    ax.set_ylim(1e-5, 1e3)
    plt.savefig(os.path.join(args.directory,'hist_voltages_norm_fit.png'))
    plt.close("all")
