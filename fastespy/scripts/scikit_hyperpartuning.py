import numpy as np
import argparse
import logging

from pathlib import PosixPath
from fastespy.io.readpydata import convert_data_to_ML_format, load_data_rikhav
from fastespy.mlscikit import hyperpartune
from fastespy.utils import init_logging

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
                        help='Use a coarse grid for hyper parameter optimization',
                        )
    parser.add_argument('--class-weight-grid', action="store_true",
                        help='Use a grid for class weight optimization',
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
    result, data, t_tot_hrs = load_data_rikhav(files, feature_names, light_cleaning_cuts=light_cleaning_cuts)

    # convert data to ML format
    X, y = convert_data_to_ML_format(result,
                                     feature_names,
                                     bkg_type=0,
                                     signal_type=1)
    # define grid and classifier
    default_pars = hyperpartune.default_pars[args.classifier]
    if args.coarse_grid:
        param_grid = hyperpartune.param_grid_coarse[args.classifier]
    else:
        param_grid = hyperpartune.param_grid[args.classifier]

    classifier = hyperpartune.clf[args.classifier](random_state=args.random_state,
                                                   **default_pars)

    hyperpartune.run_hyper_par_opt_all_folds(X, y,
                                             feature_names=feature_names,
                                             classifier=classifier,
                                             default_pars=default_pars,
                                             param_grid=param_grid,
                                             t_tot_hrs=t_tot_hrs,
                                             data=data,
                                             kfolds=args.kfolds,
                                             classifier_name=args.classifier,
                                             random_state=args.random_state,
                                             log_data=args.log_data,
                                             use_pca=args.use_pca,
                                             out_dir=out_dir,
                                             n_jobs=args.n_jobs)
