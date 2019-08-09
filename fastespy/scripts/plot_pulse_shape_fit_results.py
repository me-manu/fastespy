from __future__ import absolute_import, division, print_function
import argparse
import glob
import os
import numpy as np
from fastespy.fitting import pvalue
from fastespy.fitting import TimeLine, FitTimeLine
from fastespy.readpydata import readgraphpy
from fastespy.plotting import plot_time_line
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    usage = "usage: %(prog)s -d directory"
    description = "Plot the fit results"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-d', '--directory', required=True, help='Directory with npz data files')
    parser.add_argument('--dvdt_thr', help='Trigger threshold for derivative in mV / micro sec', type=float,
                        default = 30.)
    parser.add_argument('--v_thr', help='Trigger threshold for pulse height in V', type=float,
                        default = 0.)
    parser.add_argument('-s', '--suffix', help='suffix for data files', default='tes2')
    parser.add_argument('-i', '--usedchunk', help='the data chunk to use when mask is applied', default=0, type=int)
    args = parser.parse_args()

    # read the data
    t, v, tin, vin = readgraphpy(args.directory, prefix=args.suffix)
    t = t[args.usedchunk]
    v = v[args.usedchunk]

    # find files
    suffix = "_dvdtthr{0:.0f}_vthr{1:.0f}_c{2:n}".format(args.dvdt_thr, args.v_thr * 1e3, args.usedchunk)
    result_files = glob.glob(os.path.join(args.directory,
        'fit_results{0:s}*.npy'.format(suffix)))

    result_files = sorted(result_files, key=lambda f: int(os.path.basename(f).split('.npy')[0][-5:]))

    result = dict(tr=[], td=[], t0=[], A=[], c=[])
    err = dict(tr=[], td=[], t0=[], A=[], c=[])
    pval = []
    vavg = []

    # loop through files
    for i, rf in enumerate(result_files):
        print("loading file {0:n} / {1:n}".format(i + 1, len(result_files)))
        r = np.load(rf, allow_pickle=True).flat[0]
        # extract the rise and decay times
        # trigger times, amplitudes, and backgrounds
        if r is None:
            continue

        if r['result']['fit_ok']:
            if pvalue(r['result']['dof'], r['result']['chi2']) > 0.05:
                pval.append(pvalue(r['result']['dof'], r['result']['chi2']))

                idup = np.round(r['parameters']['stepup'] * 1e-6 * r['parameters']['fSample'], 0).astype(np.int)
                idlo = np.round(r['parameters']['steplo'] * 1e-6 * r['parameters']['fSample'], 0).astype(np.int)
                idt0 = np.where(t == r['t0'])[0][0]
                if idt0-idlo < 0:
                    idlo = 0
                if idt0+idup >= t.size:
                    idup = 0

                for kk in result.keys():
                    if kk == 't0':
                        result[kk] += \
                            [r['result']['value'][k] + r['t0'] for k in r['result']['value'].keys() if kk in k]
                        err[kk] += \
                            [r['result']['error'][k] + r['t0'] for k in r['result']['error'].keys() if kk in k]
                    else:
                        x = [r['result']['value'][k] for k in r['result']['value'].keys() if kk in k]
                        dx = [r['result']['error'][k] for k in r['result']['error'].keys() if kk in k]

                        result[kk] += x
                        err[kk] += dx

                        if kk == 'td' or kk == 'tr':
                            if np.any(np.array(x) < 1.):
                                print("sharp rise/decay time detected in trigger window {0:n}: {1}".format(i+1, x))

                                plot_time_line((t[idt0-idlo:idt0+idup] - t[idt0-idlo]) * 1e6 -
                                               r['parameters']['steplo'],
                                               v[idt0-idlo:idt0+idup] * 1e3,
                                               dv=r['parameters']['dv'],
                                               function=r['parameters']['function'],
                                               func_kwargs=r['result'])

                                fname = os.path.join(args.directory, 'fit_pulse-t-sharp'
                                                                     '_{0:05n}{1:s}.png'.format(i+1, suffix))

                                plt.savefig(fname, format='png', dpi=150)
                                print("plotted time line to {0:s}".format(fname))
                                plt.close("all")

                # if only one pulse in trigger window, add to average pulse array
                if len(r['result']['value'].keys()) == 5:
                    vavg.append(v[idt0-idlo: idt0+idup])

    # determine and fit average pulse
    if len(vavg):
        vavg = np.array(vavg) * 1e3
        print ("shape of average pulse array: {0}".format(vavg.shape))
        tavg = np.linspace(0.,vavg.shape[1] / r['parameters']['fSample'] * 1e6, vavg.shape[1])

        # fit the average pulse
        ftl = FitTimeLine(t=tavg,
                          v=vavg.mean(axis=0),
                          dv=np.sqrt(vavg.var(axis=0)),
                          fSample=r['parameters']['fSample'])
        kwargs = {}

        tr, td = 1., 10.

        if r['parameters']['function'] == 'tesresponse':
            kwargs['pinit'] = dict(
                tr_000=tr,
                td_000=td,
                c=ftl._v[0],
                A_000=np.abs(np.min(ftl.v))
            )
        elif r['parameters']['function'] == 'expflare':
            kwargs['pinit'] = dict(
                tr_000=0.5,
                td_000=7.,
                c=0.,
                A_000=np.abs(np.min(ftl.v)) * 0.66
            )

        kwargs['limits'] = dict(
            t0_000=[ftl.t.min(), ftl.t.max()],
            tr_000=[1e-5, 1e2],
            td_000=[1e-5, 1e2],
            c=[-100., 100.],
            A_000=[1e-5, 1e3]
        )

        try:
            ravg = ftl.fit(tmin=None,
                        tmax=None,
                        function=r['parameters']['function'],
                        minos=1., parscan='none',
                        dvdt_thr=-1. * args.dvdt_thr if args.dvdt_thr < 25. else -25.,
                        **kwargs)

        except (RuntimeError,ValueError) as e:
            print("Couldn't fit average pulse, Error message: {0}".format(e))
            ravg = None

        fig = plt.figure(figsize = (6, 4))
        plot_time_line(tavg, vavg.mean(axis=0), dv=np.sqrt(vavg.var(axis=0)),
                       function=r['parameters']['function'],
                       func_kwargs=ravg, data_label='average pulse', func_label='fit to average pulse')

        fig.savefig(os.path.join(args.directory, 'avg_pulse{0:s}.png'.format(suffix)),
                    dpi=150, format='png')
        plt.close("all")

    # plot histograms of single fits
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
    plt.savefig(os.path.join(args.directory, 'fit_results_plot{0:s}.png'.format(suffix)))

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
    plt.savefig(os.path.join(args.directory, 'tr_vs_td{0:s}.png'.format(suffix)))
    plt.close("all")
