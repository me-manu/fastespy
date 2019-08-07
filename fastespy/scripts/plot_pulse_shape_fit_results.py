from __future__ import absolute_import, division, print_function
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from fastespy.fitting import pvalue


if __name__ == '__main__':
    usage = "usage: %(prog)s -d directory"
    description = "Plot the fit results"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-d', '--directory', required=True, help='Directory with npz data files')
    parser.add_argument('--dvdt_thr', help='Trigger threshold for derivative in mV / micro sec', type=float,
                        default = 25.)
    args = parser.parse_args()

    # find files
    result_files = glob.glob(os.path.join(
        args.directory, 'fit_results_dvdtthr{0:.0f}_*.npy'.format(args.dvdt_thr)))
    result_files = sorted(result_files, key=lambda f: int(os.path.basename(f).split('.npy')[0][-5:]))

    result = dict(tr=[], td=[], t0=[], A=[], c=[])
    err = dict(tr=[], td=[], t0=[], A=[], c=[])
    pval = []

    # loop through files
    for i, rf in enumerate(result_files):
        print ("loading file {0:n} / {1:n}".format(i + 1, len(result_files)))
        r = np.load(rf, allow_pickle=True).flat[0]
        # extract the rise and decay times
        # trigger times, amplitudes, and backgrounds
        if r is None:
            continue

        if r['result']['fit_ok']:
            if pvalue(r['result']['dof'], r['result']['chi2']) > 0.05:
                pval.append(pvalue(r['result']['dof'], r['result']['chi2']))
                for kk in result.keys():
                    if kk == 't0':
                        result[kk] += \
                            [r['result']['value'][k] + r['t0'] for k in r['result']['value'].keys() if kk in k]
                        err[kk] += \
                            [r['result']['error'][k] + r['t0'] for k in r['result']['error'].keys() if kk in k]
                    else:
                        result[kk] += \
                            [r['result']['value'][k] for k in r['result']['value'].keys() if kk in k]
                        err[kk] += \
                            [r['result']['error'][k] for k in r['result']['error'].keys() if kk in k]

    result = {k: np.array(v) for k, v in result.items()}
    err = {k: np.array(v) for k, v in err.items()}
    pval = np.array(pval)

    # fitting function symmetric in rise and decay time
    # and sometimes they switch places.
    # correct for this:
    m = np.greater(result['tr'], result['td'])  # instances where tr and td switched places

    tr = np.zeros_like(result['tr'])
    td = np.zeros_like(result['td'])
    tr_err = np.zeros_like(result['tr'])
    td_err = np.zeros_like(result['td'])

    tr[m] = result['td'][m]
    tr[~m] = result['tr'][~m]
    tr_err[m] = err['td'][m]
    tr_err[~m] = err['tr'][~m]

    td[m] = result['tr'][m]
    td[~m] = result['td'][~m]
    td_err[m] = err['tr'][m]
    td_err[~m] = err['td'][~m]

    result['tr'] = tr
    result['td'] = td
    err['tr'] = tr_err
    err['td'] = td_err

    # check if everything went ok
    assert np.all(np.less(tr, td))

    # plot the results
    bins = dict(tr=np.linspace(0.,5.,50), td=np.linspace(0.,20.,50),
                A=np.linspace(10.,100.,50), c=np.linspace(-5.,20.,50) )

    xlabel = dict(tr='$t_\mathrm{rise}\,(\mu\mathrm{s})$', td='$t_\mathrm{decay}\,(\mu\mathrm{s})$',
                  A='Amplitude (mV)', c='Constant Background (mV)')
    unit = dict(tr='$\mu$s', td='$\mu$s', A='mV', c='mV')

    fig = plt.figure(figsize = (2*6, 2*4))
    for i, k in enumerate(['tr', 'td', 'A', 'c']):
        ax = fig.add_subplot(2,2,i+1)
        ax.hist(result[k], bins=bins[k])
        ax.set_xlabel(xlabel[k])
        med = np.median(result[k])
        ax.axvline(med, color=plt.cm.tab10(0.1), ls='--',
                   label='Median = {0:.2f}$\,${1:s}'.format(med, unit[k]))
        ax.legend(loc = 0)
    #ax = plt.subplot(2,2,4)
    #ax.hist(pval, bins=20)
    #ax.set_xlabel('$p$-value')

    plt.suptitle(args.directory)
    plt.subplots_adjust(hspace = 0.25)
    plt.savefig(os.path.join(args.directory, 'fit_results_plot.png'))

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.errorbar(result['tr'], result['td'], xerr=err['tr'], yerr=err['td'],
                 ls='none', marker='o', alpha=0.5, mec='k', zorder=0)

    ax.axvline(np.median(result['tr']), color=plt.cm.tab10(0.1), ls='--', zorder=1)
    ax.axhline(np.median(result['td']), color=plt.cm.tab10(0.1), ls='--', zorder=1)

    ax.set_xlim(bins['tr'].min()-1., bins['tr'].max()+1.)
    ax.set_ylim(bins['td'].min()-1., bins['td'].max()+1.)
    ax.set_xlabel(xlabel['tr'])
    ax.set_ylabel(xlabel['td'])
    plt.savefig(os.path.join(args.directory, 'tr_vs_td.png'))
    plt.close("all")
