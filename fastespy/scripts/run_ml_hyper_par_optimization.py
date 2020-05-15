from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import copy
import logging
from fastespy.analysis import init_logging
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

clf = dict(
    dt=DecisionTreeClassifier,
    bdt=GradientBoostingClassifier,
    rf=RandomForestClassifier
)

param_grid = dict(
    dt={'ccp_alpha': np.linspace(0., 0.002, 11),
        'min_samples_split': np.arange(2, 113, 10),
        'max_depth': np.arange(2, 11, 1)
        },
    bdt={},
    rf={}
)

default_pars = dict(
    dt={'criterion': 'gini',
        'min_samples_leaf': 1,
        },
    bdt={},
    rf={}
)

def profile_params(results, scoring, clfid):
    """
    Profile the parameters.
    For each value of a parameter, compute
    the best mean test and train scores and standard deviations
    from profiling, i.e., for each value of a parameter
    set the other grid parameters to the values that maximize the score.
    """
    mean_best_test = {}
    mean_best_train = {}
    std_best_test = {}
    std_best_train = {}

    for score in scoring.keys():

        mean_best_test[score] = {}
        mean_best_train[score] = {}
        std_best_test[score] = {}
        std_best_train[score] = {}

        for param, v in param_grid[clfid].items():

            mean_best_test[score][param] = np.zeros_like(v).astype(np.float)
            mean_best_train[score][param] = np.zeros_like(v).astype(np.float)
            std_best_test[score][param] = np.zeros_like(v).astype(np.float)
            std_best_train[score][param] = np.zeros_like(v).astype(np.float)

            for i, vi in enumerate(v):
                # create a mask where flattened param array corresponds to k = vi
                m = results[f'param_{param}'] == vi
                # get the best value for this vi
                idmax_test = np.argmax(results[f'mean_test_{score}'][m])
                idmax_train = np.argmax(results[f'mean_train_{score}'][m])

                mean_best_test[score][param][i] = results[f'mean_test_{score}'][m][idmax_test]
                std_best_test[score][param][i] = results[f'std_test_{score}'][m][idmax_test]

                mean_best_train[score][param][i] = results[f'mean_train_{score}'][m][idmax_train]
                std_best_train[score][param][i] = results[f'std_train_{score}'][m][idmax_train]

    return mean_best_test, std_best_test, mean_best_train, std_best_train


if __name__ == '__main__':
    usage = "usage: %(prog)s -i indir -o outdir "
    description = "Perform a hyper parameter optimization for machine learning on pulse fit results"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--indir', required=True, help='Path with cobmined npy files for all triggers')
    parser.add_argument('-o', '--outdir', required=True, help='Path to output file')
    parser.add_argument('-c', '--classifier', required=True, help='The classifier to be tested',
                        choices=['rf', 'dt', 'bdt'], default='dt')
    parser.add_argument('-t', '--collection_time', required=True, help='Data collection time in seconds', type=float)
    parser.add_argument('-f', '--fraction_valid',
                        help='Fraction of data set aside for final validation',
                        type=float,
                        )
    # still needs to be implemented
    #parser.add_argument('-s', '--stratify',
                        #help='keep the fraction of signal and background events in train and test data sets',
                        #type='int', default=0)
    parser.add_argument('-b', '--bkg', required=False, help='Background to train against',
                        choices=['intrinsic', 'extrinsic'], default='extrinsic')
    parser.add_argument('-k', '--kfolds', required=False, help='The number of k folds for cross validation',
                        type=int, default=5)
    args = parser.parse_args()
    init_logging("INFO", color=True)

    cla = vars(args)

    # First, we get the data path where
    # the collected results from the pulse shape fit that were read
    # in with `read_result_pulse_shape_triggered.py` are stored.

    for i, result_file in enumerate(glob.glob(os.path.join(args.indir, "*.npy"))):
        logging.info(result_file)
        r = np.load(result_file, allow_pickle=True).flat[0]
        if not i:
            result = r
        else:
            for k, v in result.items():
                result[k] = np.append(v, r[k])

    # print some stats
    logging.info("In total, the files contain {0:n} events".format(result[k].size))
    logging.info("Of which {0:n} are events taken with light".format(np.sum(result['type'] == 0)))
    logging.info("Of which {0:n} are events taken w/o light and no fiber coupled to TES (intrinsic)".format(
        np.sum(result['type'] == 1)))
    logging.info("Of which {0:n} are events taken w/o light and a fiber coupled to TES (extrinsic)".format(
        np.sum(result['type'] == 2)))

    # Build a data vector `X` which has the size `n_samples x n_features`
    features = ['integral', 'amplitude', 'tr', 'td', 'chi2_dof']
    X = np.zeros((result['type'].size, len(features)))
    y = copy.deepcopy(result['type'])

    for i, k in enumerate(features):
        X[:, i] = copy.deepcopy(result[k])

    if args.bkg == 'intrinsic':
        m = result['type'] < 2
        class_labels = ['light', 'intrinsic']
        prefix_plot = 'light_vs_int'

    elif args.bkg == 'extrinsic':
        m = (result['type'] == 0) | (result['type'] == 2)
        y[y > 1] = 1
        class_labels = ['light', 'extrinsic']
        prefix_plot = 'light_vs_extrinsic'

    # reverse labels, so that 1 = light, 0 = background
    y = np.invert(y.astype(np.bool)).astype(np.int)

    # First data split:
    # one validation set, one set for tuning hyper parameters
    X_train, X_test, y_train, y_test = train_test_split(X[m], y[m],
                                                        random_state=42,
                                                        stratify=None,
                                                        test_size=args.fraction_valid
                                                        )
    logging.info("Data shape X: {0}, y: {1}".format(X_train.shape, y_train.shape))


    # With the test data, perform a K-fold cross validation to
    # find the best hyper parameters.
    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)

    # scoring: set the scoring parameter for optimzation
    # see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # for implemented defaults
    # scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'Recall': 'recall'}

    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}
    refit = 'AUC'

    t0 = time.time()
    logging.info("Starting parameter grid search... ")
    gs = GridSearchCV(clf[args.classifier](random_state=42, **default_pars[args.classifier]),
                      param_grid=param_grid[args.classifier],
                      scoring=scoring,
                      refit=refit,
                      return_train_score=True,
                      cv=kf
                      )
    gs.fit(X_train, y_train)
    t1 = time.time()
    logging.info("The parameter search took {0:.2f} s".format(t1-t0))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # profiling results: mean and standard deviation for testing and training
    mt, st, mtr, sstr = profile_params(gs.cv_results_, scoring, args.classifier)

    # TODO: for each scoring function, calculate
    # TODO: the score of the best classifier on the test data
    # TODO: using
    #for v in scoring.values():
        #scorer = get_scorer(v)
        #scorer(gs.best_estimator_, X_test, y_test)
    # TODO: save results
