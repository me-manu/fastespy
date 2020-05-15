from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import logging
from fastespy.analysis import init_logging
from fastespy.functions import TimeLine, calc_xi
import glob
from tqdm import tqdm

if __name__ == '__main__':
    usage = "usage: %(prog)s -i indir -o outdir "
    description = "Read in the the pulse shape fit results for triggered data files"\
                  " created with run_pulse_shape_fit_trigger.py script"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--indir', required=True, help='Path with npy fit result functions')
    parser.add_argument('-o', '--outfile', required=True, help='Path to output file')
    parser.add_argument('-t', '--type', required=True,
                        help='Type of measurement: 0: light, 1: intrinsic, 2: extrinsic',
                        type=int,
                        choices=[0, 1, 2])
    args = parser.parse_args()
    init_logging("INFO", color=True)

    fit_result_files = glob.glob(os.path.join(args.indir, "*.npy"))
    # sort the result files by trigger window
    fit_result_files = sorted(fit_result_files,
                              key=lambda f: int(os.path.basename(f).split("_")[-2])
                              )

    logging.info("Found {0:d} fit result files".format(len(fit_result_files)))

    logging.info("Starting loop to read in results")

    integral = np.array([])
    tr = np.array([])
    td = np.array([])
    amplitude = np.array([])
    chi2 = np.array([])
    dof = np.array([])
    trigger_found = np.array([])
    trigger_id = np.array([])
    fit_ok = np.array([])

    for fit_result_file in tqdm(fit_result_files):
        r = np.load(fit_result_file, allow_pickle=True).flat[0]

        trigger_id = np.append(trigger_id, int(os.path.basename(fit_result_file).split("_")[-2])).astype(np.int)

        # no trigger found
        if len(r['result']) == 1:
            trigger_found = np.append(trigger_found, False).astype(np.bool)
            continue
        else:
            trigger_found = np.append(trigger_found, True).astype(np.bool)

        # calculate the integral over the pulse shapes
        tl = TimeLine.read_fit_result(r)
        # max time for integration:
        # the length of the trigger window in mu s
        # multiplied by 2 in case that second pulse occurred at the end of the
        # trigger window
        tmax = (r['result']['dof'] + r['result']['npar']) / r['parameters']['fSample'] * 1e6 * 2

        # this is now an array with the intgral for each pulse component
        integral_this = tl.integral(tmin=0., tmax=tmax, tstep=2000, **r['result']['value'])
        integral = np.append(integral, integral_this)

        # check whether rise time is really rise time or is flipped with decay time
        for i in list(range(tl.numcomp)):
            tr_this = r['result']['value']['tr_{0:03n}'.format(i)]
            td_this = r['result']['value']['td_{0:03n}'.format(i)]
            if r['parameters']['function'] == 'tesresponse':
                if calc_xi(td=td_this, tr=tr_this) <= 0.:
                    tr = np.append(tr, tr_this)
                    td = np.append(td, td_this)
                else:
                    tr = np.append(tr, td_this)
                    td = np.append(td, tr_this)
            else:
                tr = np.append(tr, tr_this)
                td = np.append(td, td_this)
            amplitude = np.append(amplitude, r['result']['value']['A_{0:03n}'.format(i)])
            chi2 = np.append(chi2, r['result']['chi2'])
            dof = np.append(dof, r['result']['dof'])
            fit_ok = np.append(fit_ok, r['result']['fit_ok']).astype(np.bool)

    logging.info("Trigger was not found in {0:d} / {1:d} file(s)".format(
                    np.sum(~trigger_found), len(fit_result_files)
                    ))

    if np.any(trigger_found == False):
        logging.info("Printing trigger ids where no trigger was found:\n{0}".format(trigger_id[~trigger_found]))

    logging.info("iminuit fit status was not ok for {0:d} / {1:d} pulse profiles".format(
        np.sum(~fit_ok), fit_ok.size
    ))

    # convert result to dict:
    result = dict(
        integral=integral,
        amplitude=amplitude,
        tr=tr,
        td=td,
        chi2=chi2,
        dof=dof,
        chi2_dof=chi2 / dof,
        fit_ok=fit_ok,
        type=np.ones_like(amplitude).astype(np.int) * args.type,
        function=r['parameters']['function']
    )

    logging.info("saving results to {0:s}".format(args.outfile))
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(args.outfile, result)

