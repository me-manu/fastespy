import numpy as np
import argparse
import logging
import tqdm

from pathlib import PosixPath
from sklearn.model_selection import StratifiedKFold
from fastespy.readpydata import convert_data_to_ML_format
from fastespy.ml import MLHyperParTuning
from fastespy import feldman_cousins as fc
from fastespy.analysis import init_logging


def load_data(files, feature_names, light_cleaning_cuts={}):
    """
    Load data from Rikhav's pulse fitting

    Parameters
    ----------
    files: list
        list of file names

    feature_names: list
        list of feature names to be used

    light_cleaning_cuts: dict
        dictionary with feature names as key and cleaning options to be used

    Returns
    -------
    Tuple containing:
        - Dictionary with fitting data
        - Dictionary for raw data
        - Float with total observation time
    """

    result = {'type': []}
    t_tot_hrs = 0.
    id_rejected = []
    data = {"time": [], "data": []}
    # loop through files
    logging.info("Reading data")

    for f in tqdm.tqdm(files):
        x = np.load(f, allow_pickle=True).tolist()

        # for each file: calculate observation time
        t_start = 1e10
        t_stop = 0.

        if 'light' in str(f):
            id_rejected.append([])

        # loop through triggers
        for i in range(1, len(x.keys()) + 1):
            # light sample cleaning
            if 'light' in str(f):
                m = True
                for c, v in light_cleaning_cuts.items():
                    # print(i, v, {c.split()[0]: x[i][c]})
                    m &= eval(v, {c.split()[0]: x[i][c]})

                if not m:
                    id_rejected[-1].append(i)
                    continue

            for name in feature_names:
                if not name in result.keys():
                    result[name] = []

                result[name].append(x[i][name])

            # save raw data
            data['time'].append(x[i]['time'])
            data['data'].append(x[i]['data'])

            if 'intrinsic' in str(f):
                if x[i]['end time in hrs'] > t_stop:
                    t_stop = x[i]['end time in hrs']
                if x[i]['start time in hrs'] < t_start:
                    t_start = x[i]['start time in hrs']
                result['type'].append(0)

            if 'light' in str(f):
                result['type'].append(1)

        if 'intrinsic' in str(f):
            t_tot_hrs += t_stop - t_start  # only add for dark count rate
    for rej in id_rejected:
        logging.info("Rejected {0:n} triggers in light file".format(len(rej)))
    for k in ['time', 'data']:
        data[k] = np.array(data[k])
    # convert into into numpy arrays
    for k, v in result.items():
        if k == 'type':
            dtype = np.bool
        else:
            dtype = np.float32
        result[k] = np.array(v, dtype=dtype)

    logging.info("In total, there are {0:n} light events and {1:n} background events"
                 " for an observation time of {2:.2f} hours".format(result['type'].sum(),
                                                                    np.invert(result['type']).sum(),
                                                                    t_tot_hrs
                                                                    ))
    return result, data, t_tot_hrs


if __name__ == "__main__":
    usage = "usage: %(prog)s -i indir -o outdir "
    description = "Perform a hyper parameter optimization for machine learning on pulse fit results"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--indir', required=True, help='Path with npy files for all triggers')
    parser.add_argument('-o', '--outdir', required=True, help='Path to output file')
    parser.add_argument('-c', '--classifier', required=True, help='The classifier to be tested',
                        choices=['rf', 'dt', 'bdt', 'mlp'], default='dt')
    parser.add_argument('--n_jobs', help='Number of processors to use', type=int, default=1)
    parser.add_argument('--random_state', help='Random state', type=int, default=42)
    parser.add_argument('--coarse-grid', action="store_true",
                        help='Use a coarse grid for hyper parameter optimazation',
                        )
    parser.add_argument('--log-data', action="store_true",
                        help='Transform data into log space',
                        )
    parser.add_argument('--use-pca', action="store_true",
                        help='Transform data using PCA',
                        )
    parser.add_argument('-k', '--kfolds', required=False, help='The number of k folds for cross validation',
                        type=int, default=5)

    args = parser.parse_args()
    init_logging("INFO", color=True)

    # get the files
    in_dir = PosixPath(args.indir)
    out_dir = PosixPath(args.outdir)

    files = list(in_dir.glob("*.npy"))

    logging.info("Using files:")
    for f in files:
        logging.info(f)

    # define the feature names
    feature_names = []
    remove = ['data', 'time', 'pulse integral raw', 'voltage error',
              'error', 'start time in hrs', 'end time in hrs',
              'trigger time'
              ]

    x = np.load(files[0], allow_pickle=True).tolist()
    for k in x[1].keys():
        if not k in remove and not 'error' in k:
            feature_names.append(k)

    logging.info("Using features names {}".format(feature_names))

    # define light cleaning cuts
    light_cleaning_cuts = {
        "chi2 reduced": "chi2 < 6.",
        "decay time": "decay < 10.e-6",
        "trigger time": "(trigger >= 29.5e-6) & (trigger <= 30.8e-6)",  # from gaussian fit, 5 sigma interval
    }
    logging.info("Using cleaning cuts {}".format(light_cleaning_cuts))

    # read the data
    result, data, t_tot_hrs = load_data(files, feature_names, light_cleaning_cuts=light_cleaning_cuts)

    # convert data to ML format
    X, y = convert_data_to_ML_format(result,
                                     feature_names,
                                     bkg_type=0,
                                     signal_type=1)
    if args.log_data:
        logging.info("Transforming data to log space")
        X, y = MLHyperParTuning.transform_data_log(X, y, feature_names)

    # perform stratified K fold
    skf = StratifiedKFold(n_splits=args.kfolds, random_state=args.random_state, shuffle=True)
    skf.get_n_splits(X, y)

    # perform the hyper parameter optimazition
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        logging.info(" Running optimization for split {0:n} / {1:n}".format(i + 1, args.kfolds))

        ml = MLHyperParTuning(X[train_idx], y[train_idx],
                              X_test=X[test_idx],
                              y_test=y[test_idx],
                              idx_test=test_idx,
                              valid_fraction=1. / args.kfolds,
                              stratify=True,
                              random_state=args.random_state,
                              n_splits=args.kfolds)
        ml.scale_data()

        if args.use_pca:
            logging.info("Transforming data using PCA")
            ml.transform_data_pca()

        ml.make_sig_scorer(t_obs=t_tot_hrs * 3600.)

        ml.perform_grid_search(classifier=args.classifier,
                               refit='Significance',
                               coarse_grid=args.coarse_grid,
                               n_jobs=args.n_jobs)

        out_path = out_dir / "{0:05n}".format(i + 1)
        if not out_path.exists():
            out_path.mkdir(parents=True)

        # output plots
        ml.plot_confusion_matrix(ml.results, ml.scoring, path=out_path, classifier=ml.classifier)
        ml.plot_learning_curve(ml.results, path=out_path, classifier=ml.classifier)

        # generate results dict
        results = ml.make_result_dict()

        ml.plot_misidentified_time_lines(results,
                                         scorer="Significance",
                                         data_time=data['time'],
                                         data_voltage=data['data'],
                                         X=X,
                                         feature_names=feature_names,
                                         path=out_path
                                         )

        # output performance
        logging.info("Printing performance:")
        ml.print_performance_report(ml.results, ml.scoring, ml.t_obs)

        # Add Feldman & Cousins confidence interval for dark current rate
        logging.info("Running Feldman & Cousins confidence interval estimation for dark current")
        n_b = 0
        n_obs = np.arange(ml.results['confusion_matrix_test']['Significance'][0, 1] * 5)
        mus = np.linspace(0, ml.results['confusion_matrix_test']['Significance'][0, 1] * 3, 2401)
        alpha = 0.9

        lower_limits_mu, upper_limits_mu = fc.poissonian_feldman_cousins_interval(
            n_obs=n_obs,
            n_b=n_b,
            mus=mus,
            alpha=alpha,
            fix_discrete_n_pathology=False,
            n_jobs=args.n_jobs)

        lower_limits = lower_limits_mu[:, 0]
        upper_limits = upper_limits_mu[:, 0]

        # add F&C result to result dict
        results['dark_current'] = {}
        for k in ml.scoring.keys():
            results['dark_current'][k] = np.array([
                lower_limits[ml.results['confusion_matrix_test']['Significance'][0, 1]] * (
                            ml.y_test.size + ml.y_train.size) / ml.y_test.size,
                ml.results['bkg_pred_test']['Significance'],
                upper_limits[ml.results['confusion_matrix_test']['Significance'][0, 1]] * (
                            ml.y_test.size + ml.y_train.size) / ml.y_test.size
            ]) / ml.t_obs
            logging.info("dark current {0:s}: {1}".format(k, results['dark_current'][k]))

        np.save(out_path / f"r{ml.classifier}_cleaned_reduced.npy", results)

