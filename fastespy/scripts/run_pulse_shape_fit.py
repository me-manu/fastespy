from __future__ import absolute_import, division, print_function
import argparse
import glob
import os
import numpy as np
import logging
import time
from fastespy.readpydata import read_graph_py
from fastespy.analysis import build_trigger_windows, filter
from fastespy.fitting import FitTimeLine
from fastespy.analysis import init_logging
from fastespy.plotting import plot_time_line

if __name__ == '__main__':
    usage = "usage: %(prog)s -d directory -f Sampling frequency [-s suffix]"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-d', '--directory', required=True, help='Directory with npz data files')
    parser.add_argument('-f', '--fSample', required=True, help='The used sampling frequency in Hz', type=float)
    parser.add_argument('-s', '--suffix', help='suffix for data files', default='tes2')
    parser.add_argument('-i', '--usedchunk', help='the data chunk to use when mask is applied', default=0, type=int)
    parser.add_argument('-m', '--maskvalue', help='voltage values above which values are used',
                type=float, default=0.)
    parser.add_argument('--dvdt_thr', help='Trigger threshold for derivative in mV / micro sec', type=float,
                        default = 25.)
    parser.add_argument('--v_thr', help='Trigger threshold for pulse height in V', type=float,
                        default = 0.)
    parser.add_argument('--dv', help='Assumed constant readout error in mV', type=float,
                        default = 3.)
    parser.add_argument('--steplo', help='Time in micro s to be included before trigger', type=float,
                        default = 5.)
    parser.add_argument('--stepup', help='Time in micro s to be included after trigger', type=float,
                        default = 30.)
    parser.add_argument('--fmax', help='Maximum frequency in Hz above which derivative is smoothed', type=float,
                        default = 1.e6)
    parser.add_argument('--function', help='The function used for the pulse', choices=['expflare', 'tesresponse'],
                        default = 'tesresponse')
    parser.add_argument('--control_plots', help='Generate control plots', type=int,
                        default = 1)
    parser.add_argument('--filter', help='If True, fit data filtered with low pass', type=int,
                        default = 0)
    parser.add_argument('--norder', help="Filter order for derivative. If 0, don't use the filter", type=int,
                        default = 3)
    parser.add_argument('--istart', help='Start at this trigger window', type=int)
    parser.add_argument('--istop', help='Stop this trigger window', type=int)
    parser.add_argument('--minos', help='Compute minos error within this confidence level', type=float, default=1.)
    parser.add_argument('--maxcomp', help='Maximum number of components that are tested', type=int, default=3)

    args = parser.parse_args()
    init_logging("INFO", color=True)

    if not len(glob.glob(os.path.join(args.directory, "{0:s}*.root".format(args.suffix)))) == len(
            glob.glob(os.path.join(args.directory, "{0:s}*.npz".format(args.suffix)))):
        raise IOError("Error: convert files first!")

    t, v, tin, vin = read_graph_py(args.directory, prefix=args.suffix)

    # set the mask for the voltage
    if args.control_plots:

        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        for i, vv in enumerate(vin):
            plt.hist(vv, bins=100, density=True, alpha=0.5)
            plt.xlabel('Voltage (V)')
            plt.savefig(os.path.join(args.directory,'hist.png'))
            plt.close("all")

    m = []
    if len(vin):
        for i, vv in enumerate(vin):
            m.append(vv > args.maskvalue)
    else:
        for i, vv in enumerate(v):
            m.append(np.ones(vv.size, dtype=np.bool))

    # find where the first data chunk ends
    # this can happen from Ibox biasing
    where = np.where(m[args.usedchunk] == False)
    if len(where[0]):
        idxmax = where[0][0]
    else:
        idxmax = m[args.usedchunk].size

    logging.info("Using first {0:n} data points".format(idxmax))
    sli = slice(0, idxmax)

    t = t[args.usedchunk][m[args.usedchunk]][sli]
    v = v[args.usedchunk][m[args.usedchunk]][sli]

    print ("Measurement time: {0:.5f} seconds".format(t.max() - t.min()))

    # build the trigger windows using derivative
    t0, t_trig, v_trig = build_trigger_windows(t, v, fSample=args.fSample,
                                               thr=-1. * args.dvdt_thr,
                                               thr_v=-1. * args.v_thr,
                                               tstepup=args.stepup, tsteplo=args.steplo,
                                               fmax=args.fmax, norder=args.norder)

    tr = 2.  # initial guess for rise time in micro sec
    td = 4.  # initial guess for decay time in micro sec

    logging.info("Starting to fit {0:n} trigger windows".format(len(t_trig)))
    t1 = time.time()

    for i, t in enumerate(t_trig):

        if args.istart is not None:
            if i < args.istart - 1:
                continue
        if args.istop is not None:
            if i >= args.istop - 1:
                break

        print("===== Fitting Trigger window {0:n} / {1:n} ===== ".format(i+1, len(t_trig)))

        if args.filter:
            y = filter(v_trig[i], fSample=args.fSample, fmax=args.fmax, norder=args.norder) * 1e3
        else:
            y = v_trig[i] * 1e3
        dy = np.full(y.size, args.dv)

        ftl = FitTimeLine(t=(t - t0[i]) * 1e6,
                          v=y,
                          dv=dy,
                          fSample=args.fSample)

        # set the initial conditions
        kwargs = {}

        if args.function == 'tesresponse':
            kwargs['pinit'] = dict(
                tr_000=tr,
                td_000=td,
                c=ftl._v[0],
                A_000=np.abs(np.min(ftl.v))
            )
        elif args.function == 'expflare':
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
            r = ftl.fit(tmin=None,
                    tmax=None,
                    function=args.function,
                    minos=args.minos, parscan='none',
                    maxcomp=args.maxcomp,
                    dvdt_thr=-1. * args.dvdt_thr if args.dvdt_thr < 25. else -25.,
                    **kwargs)

            if args.control_plots:
                plot_time_line((t - t0[i]) * 1e6,
                               y,
                               dv=dy,
                               function=args.function,
                               func_kwargs=r)

                for j in range(r['numcomp']):
                    plt.axvline(r['value']['t0_{0:03}'.format(j)],
                                ls='--', color='r', lw=0.5)

        except (RuntimeError,ValueError) as e:
            logging.error("Couldn't fit trigger window {0:n}, Error message: {1}".format(i+1,e))
            r = {'fit_ok': False}
            if args.control_plots:
                plot_time_line((t - t0[i]) * 1e6,
                               y,
                               dv=dy)

        if args.control_plots:
            plt.savefig(os.path.join(args.directory, "fit_pulse_{0:05n}_c{1:n}.png".format(i+1, args.usedchunk)),
                        format='png', dpi=150)
            plt.close("all")

        d = dict(result=r, t0=t0[i], parameters=vars(args))
        fname = os.path.join(args.directory, 'fit_results_dvdtthr{1:.0f}_vthr{3:.0f}_c{2:n}_{0:05}.npy'.format(
            i+1,args.dvdt_thr, args.usedchunk, args.v_thr * 1e3))

        np.save(fname,d)

        logging.info("written results to {0:s}".format(fname))

        del ftl

    logging.info("Done fitting, it took {0:.2f} seconds".format(time.time() - t1))
