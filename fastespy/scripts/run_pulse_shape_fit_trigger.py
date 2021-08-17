from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import logging
import time
from fastespy.readpydata import read_converted_pickle
from fastespy.analysis import build_trigger_windows, filter
from fastespy.fitting import FitTimeLine
from fastespy.analysis import init_logging
from fastespy.plotting import plot_time_line_and_derivative
import matplotlib.pyplot as plt


def run_fit(t, y, dy, fSample, t0, args, tr=2., td=4.):
    """
    Run the time line fit

    :param t: array-like
        time values
    :param y: array-like
        voltage values
    :param dy: array-like
        uncertainty on y
    :param fSample: float
        sampling frequency
    :param t0: float
        trigger time
    :param args: argparse config parser
        additinoal arguments
    :param tr: float
        initial guess for rise time in mu s
    :param td: float
        initial guess for decay time in mu s
    :return: dict
        dictionary with fit results
    """
    ftl = FitTimeLine(t=(t - t0) * 1e6,
                      v=y,
                      dv=dy,
                      fSample=fSample)
    # set the initial conditions
    kwargs = {}
    if 'tesresponse' in args.function:
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
        result = ftl.fit(tmin=None,
                         tmax=None,
                         function=args.function,
                         minos=args.minos, parscan='none',
                         maxcomp=args.maxcomp,
                         dvdt_thr=-1. * args.dvdt_thr if args.dvdt_thr < 25. else -25.,
                         v_thr=-1. * args.v_thr,
                         **kwargs)

        if args.control_plots:
            fig, ax_t_vs_v, ax_dvdt = plot_time_line_and_derivative(
                (t - t0) * 1e6,
                y,
                dv=dy,
                function=args.function,
                func_kwargs=result,
                fSample=fSample,
                )

            for j in range(result['numcomp']):
                ax_t_vs_v.axvline(result['value']['t0_{0:03}'.format(j)],
                                  ls='--', color='r', lw=0.5)
                ax_dvdt.axvline(result['value']['t0_{0:03}'.format(j)],
                                ls='--', color='r', lw=0.5)

    except (RuntimeError, ValueError) as e:
        logging.error("Couldn't fit trigger window {0:n}-{1:n}, Error message: {2}".format(idata+1, i+1, e))
        result = {'fit_ok': False}
        if args.control_plots:
            fig, ax, _ = plot_time_line_and_derivative(
                                                       (t - t0) * 1e6,
                                                       y,
                                                       dv=dy,
                                                       fSample=fSample,
                                                       )

            if "No triggers" in str(e):
                ax.annotate("No trigger found!", xy=(0.95, 0.05), ha='right', xycoords="axes fraction")

    if args.control_plots:
        fig.savefig(os.path.join(outdir, "fit_trigger_pulse_{0:05n}_{1:05n}.png".format(idata+1, i+1)),
                    format='png', dpi=150)
        plt.close("all")

    return result


if __name__ == '__main__':
    usage = "usage: %(prog)s -i infile -o outdir "
    description = "Run the pulse shape fit on (triggered) data files converted from root to python with" \
                  "convert_root_file_to_python.py script"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--infile', required=True, help='File with tiggers in pickle.bz2 format')
    parser.add_argument('-o', '--outdir', required=True, help='Path to out directory')
    parser.add_argument('--dvdt_thr', help='Trigger threshold for derivative in mV / micro sec', type=float,
                        default = 25.)
    parser.add_argument('--v_thr', help='Trigger threshold for pulse height in V', type=float,
                        default = 0.001)
    parser.add_argument('--dv', help='Assumed constant readout error in mV', type=float,
                        default = 3.)
    parser.add_argument('--steplo', help='Time in micro s to be included before trigger, '\
                        'if not given, use full trigger window',
                        type=float)
    parser.add_argument('--stepup', help='Time in micro s to be included after trigger, '\
                        'if not given, use full trigger window',
                        type=float)
    parser.add_argument('--fmax', help='Maximum frequency in Hz above which derivative is smoothed', type=float,
                        default = 1.e6)
    parser.add_argument('--function', help='The function used for the pulse',
                        choices=['expflare', 'tesresponse', 'tesresponse_simple'],
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
    parser.add_argument('--minos', help='Compute minos error within this confidence level', type=float, default=1.)
    parser.add_argument('--maxcomp', help='Maximum number of components that are tested', type=int, default=3)

    args = parser.parse_args()
    init_logging("INFO", color=True)

    data = read_converted_pickle(args.infile, channel=args.channel)

    print("There are {0:n} trigger events in data file".format(len(data)))

    outdir = os.path.join(os.path.dirname(args.outdir),
                          "dvdtthr{0:.0f}_vthr{1:.0f}_steplo{2}_stepup{3}".format(
                          args.dvdt_thr,args.v_thr * 1e3, args.steplo, args.stepup))

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

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
            logging.info("Starting to fit {0:n} trigger windows".format(len(t_trig)))
        else:

            logging.info("===== Fitting Trigger window {0:n} / {1:n} ===== ".format(idata+1, len(data)))

            t_trig = [d['time']]
            v_trig = [d['data']]
            t0 = [d['timeStamp']]

            # shorten each trigger window if args.steplo and args.stepup are given
            if args.steplo is not None:
                m = [ t_trig[i]-t0[i] >= args.steplo/1e6 for i in range(len(t_trig))]
            else:
                m = [ np.ones(t_trig[i].size, dtype=np.bool) for i in range(len(t_trig))]

            if args.stepup is not None:
                m = [ m[i] & (t_trig[i]-t0[i] <= args.stepup/1e6) for i in range(len(t_trig))]

            if args.istart is not None:
                if idata < args.istart - 1:
                    continue
            if args.istop is not None:
                if idata >= args.istop - 1:
                    break

            t_trig = [t_trig[i][m[i]] for i in range(len(t_trig))]
            v_trig = [v_trig[i][m[i]] for i in range(len(t_trig))]

        t1 = time.time()

        for i, t in enumerate(t_trig):

            if args.buildtrigger:
                if args.istart is not None:
                    if i < args.istart - 1:
                        continue
                if args.istop is not None:
                    if i >= args.istop - 1:
                        break

                logging.info("===== Fitting Trigger window {0:n} / {1:n} ===== ".format(i+1, len(t_trig)))

            if args.filter:
                y = filter(v_trig[i], fSample=fSample, fmax=args.fmax, norder=args.norder) * 1e3
            else:
                y = v_trig[i] * 1e3
            dy = np.full(y.size, args.dv)

            result = run_fit(t, y, dy, fSample, t0[i], args, tr=2., td=4.)

            parameters = vars(args)
            parameters['fSample'] = fSample

            d = dict(result=result, t0=t0[i], parameters=parameters)
            fname = os.path.join(outdir, 'fit_trigger_results_{0:05n}_{1:05n}.npy'.format(idata+1, i+1))

            np.save(fname, d)

            logging.info("written results to {0:s}".format(fname))

    logging.info("Done with fitting")