from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import logging
import time
from fastespy.readpydata import readconvertedpickle
from fastespy.analysis import build_trigger_windows, filter
from fastespy.fitting import FitTimeLine
from fastespy.analysis import init_logging
from fastespy.plotting import plot_time_line
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    usage = "usage: %(prog)s -i infile -f Sampling frequency [-s suffix]"
    description = "Run the pulse shape fit on (triggered) data files converted from root to python with" \
                  "convert_root_file_to_pyhton.py script"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--infile', required=True, help='File with tiggers in pickle.bz2 format')
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
    parser.add_argument('--buildtrigger', help='build own trigger windows'
                                               '(i.e. this is a long-term measurement)', type=int, default=0)
    parser.add_argument('--channel', help='Alazar card channel used', type=int, default=0)

    args = parser.parse_args()
    init_logging("WARNING", color=True)

    data = readconvertedpickle(args.infile, channel=args.channel)

    print("There are {0:n} trigger events in data file".format(len(data)))

    outdir = os.path.join(os.path.dirname(args.infile),
                          "dvdtthr{0:.0f}_vthr{1:.0f}".format(args.dvdt_thr,args.v_thr * 1e3))

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # loop through trigger events
    for idata, d in enumerate(data):

        fSample = d['samplingFreq']

        if args.buildtrigger:
            t0, t_trig, v_trig = build_trigger_windows(d['time'], d['data'],
                                                       fSample=fSample,
                                                       thr=-1. * args.dvdt_thr,
                                                       thr_v=-1. * args.v_thr,
                                                       tstepup=args.stepup, tsteplo=args.steplo,
                                                       fmax=args.fmax, norder=args.norder)
            print("Starting to fit {0:n} trigger windows".format(len(t_trig)))
        else:

            print("===== Fitting Trigger window {0:n} / {1:n} ===== ".format(idata+1, len(data)))

            t_trig = [d['time']]
            v_trig = [d['data']]
            t0 = [d['timeStamp']]
            if args.istart is not None:
                if idata < args.istart - 1:
                    continue
            if args.istop is not None:
                if idata >= args.istop - 1:
                    break

        tr = 2.  # initial guess for rise time in micro sec
        td = 4.  # initial guess for decay time in micro sec

        t1 = time.time()

        for i, t in enumerate(t_trig):

            if args.buildtrigger:
                if args.istart is not None:
                    if i < args.istart - 1:
                        continue
                if args.istop is not None:
                    if i >= args.istop - 1:
                        break

                print("===== Fitting Trigger window {0:n} / {1:n} ===== ".format(i+1, len(t_trig)))

            if args.filter:
                y = filter(v_trig[i], fSample=fSample, fmax=args.fmax, norder=args.norder) * 1e3
            else:
                y = v_trig[i] * 1e3
            dy = np.full(y.size, args.dv)

            ftl = FitTimeLine(t=(t - t0[i]) * 1e6,
                              v=y,
                              dv=dy,
                              fSample=fSample)

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
                            minos=1., parscan='none',
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

            except (RuntimeError, ValueError) as e:
                logging.error("Couldn't fit trigger window {0:n}-{1:n}, Error message: {2}".format(idata+1, i+1, e))
                r = {'fit_ok': False}
                if args.control_plots:
                    plot_time_line((t - t0[i]) * 1e6,
                                   y,
                                   dv=dy)

                    if "No triggers" in str(e):
                        plt.annotate("No trigger found!", xy=(0.95, 0.05), ha='right', xycoords="axes fraction")

            if args.control_plots:
                plt.savefig(os.path.join(outdir, "fit_trigger_pulse_{0:05n}_{1:05n}.png".format(idata+1, i+1)),
                            format='png', dpi=150)
                plt.close("all")

            parameters = vars(args)
            parameters['fSample'] = fSample

            d = dict(result=r, t0=t0[i], parameters=parameters)
            fname = os.path.join(outdir, 'fit_trigger_results_{0:05n}_{1:05n}.npy'.format(idata+1, i+1))

            np.save(fname, d)

            print("written results to {0:s}".format(fname))

            del ftl

    logging.info("Done with fitting")