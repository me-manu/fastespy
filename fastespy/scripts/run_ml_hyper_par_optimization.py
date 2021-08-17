from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import copy
import logging
from fastespy.analysis import init_logging
from fastespy.readpydata import read_fit_results_manuel, read_fit_results_rikhav, read_fit_results_axel
from fastespy.readpydata import convert_data_to_ML_format
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import get_scorer, make_scorer, fbeta_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = dict(
    dt=DecisionTreeClassifier,
    bdt=GradientBoostingClassifier,
    rf=RandomForestClassifier,
    mlp=MLPClassifier
)

param_grid = dict(
    dt={'ccp_alpha': np.linspace(0., 0.002, 11),
        'min_samples_split': np.arange(2, 113, 10),
        'max_depth': np.arange(2, 11, 1)
        },
    bdt={'n_estimators': np.arange(100, 1100, 100),
         'learning_rate': np.arange(0.1, 1.1, 0.1),
         'max_depth': np.arange(2, 6, 1)
         },
    rf={'n_estimators': np.arange(100, 600, 100),
        'max_features': np.arange(1, 6, 1),
        'min_samples_split': np.arange(2, 82, 10),
        },
    mlp={
        'hidden_layer_sizes': ((10,), (50,), (10, 10), (50, 50), (10, 10, 10), (50, 50, 50)),
        'alpha': 10.**np.arange(-4, 0.5, 0.5)
        }
)

default_pars = dict(
    dt={'criterion': 'gini',
        'min_samples_leaf': 1,
        },
    bdt={'loss': 'deviance',
         'min_samples_split': 2
         },
    rf={'criterion': 'gini',
        'max_depth': None,  # fully grown trees
        },
    mlp={'learning_rate' : 'constant',
         'activation': 'relu',
         'max_iter': 1000,
         'solver': 'adam',
         'shuffle': True,
         'tol': 1e-4
         }
)


if __name__ == '__main__':
    usage = "usage: %(prog)s -i indir -o outdir "
    description = "Perform a hyper parameter optimization for machine learning on pulse fit results"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--indir', required=True, help='Path with combined npy files for all triggers')
    parser.add_argument('-o', '--outdir', required=True, help='Path to output file')
    parser.add_argument('-c', '--classifier', required=True, help='The classifier to be tested',
                        choices=['rf', 'dt', 'bdt', 'mlp'], default='dt')
    parser.add_argument('--n_jobs', help='Number of processors to use', type=int, default=1)
    parser.add_argument('--fit_type', help='Type of fit performed', default='Manuel',
                        choices=['Manuel', 'Rikhav', 'Axel'])
    parser.add_argument('-f', '--fraction_valid',
                        help='Fraction of data set aside for final validation',
                        type=float,
                        default=0.2
                        )
    # still needs to be implemented
    #parser.add_argument('-s', '--stratify',
                        #help='keep the fraction of signal and background events in train and test data sets',
                        #type='int', default=0)
    parser.add_argument('-b', '--bkg_id', required=True, help='ID of Background to train against', type=int)
    parser.add_argument('-k', '--kfolds', required=False, help='The number of k folds for cross validation',
                        type=int, default=5)
    args = parser.parse_args()
    init_logging("INFO", color=True)

    cla = vars(args)

    if args.fit_type == 'Manuel':
        result = read_fit_results_manuel(args.indir)
        features = ['integral', 'amplitude', 'tr', 'td', 'chi2_dof', 'const']
    elif args.fit_type == 'Rikhav':
        result = read_fit_results_rikhav(args.indir)
        #features = ['pulse_integral_fit',
                    #'amplitude', 'rise_time', 'decay_time', 'chi2_reduced']
        # use my integral
        features = ['integral',
                    'amplitude', 'rise_time', 'decay_time', 'chi2_reduced']
        signal_id = 3
    elif args.fit_type == 'Axel':
        result = read_fit_results_axel(args.indir)
        features = ['peak', 'decay', 'chi2', 'integral', 'ampli', 'const']
        signal_id = 1

    # Build a data vector `X` which has the size `n_samples x n_features`
    # and corresponding class vector.
    # signal corresponds to class 1, bkg to class 0

    X, y = convert_data_to_ML_format(result, features,
                                     bkg_type=args.bkg_id,
                                     signal_type=signal_id)

    suffix = "{0:s}_{1:s}_bkg{2:n}".format(args.fit_type,
                                           args.classifier,
                                           args.bkg_id)

    logging.info(f"suffix for output files: {suffix}")


    # First data split:
    # one validation set, one set for tuning hyper parameters
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        stratify=None,
                                                        test_size=args.fraction_valid
                                                        )
    logging.info("Data shape X: {0}, y: {1}".format(X_train.shape, y_train.shape))

    # With the test data, perform a K-fold cross validation to
    # find the best hyper parameters.
    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)

    # scoring: set the scoring parameter for optimization
    # see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # for implemented defaults
    # scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'Recall': 'recall'}

    scoring = {'AUC': 'roc_auc',
               'Accuracy': 'accuracy',
               'Precision': 'precision',
               'Recall': 'recall',
               'F_1': 'f1',
               'F_2': make_scorer(fbeta_score, beta=2),
               'F_{1/2}': make_scorer(fbeta_score, beta=0.5),
               }

    #refit = 'AUC'
    refit = 'Accuracy'

    t0 = time.time()
    logging.info("Starting parameter grid search... ")
    gs = GridSearchCV(clf[args.classifier](random_state=42, **default_pars[args.classifier]),
                      param_grid=param_grid[args.classifier],
                      scoring=scoring,
                      refit=refit,
                      return_train_score=True,
                      cv=kf,
                      verbose=1,
                      n_jobs=args.n_jobs
                      )
    gs.fit(X_train, y_train)
    t1 = time.time()
    logging.info("The parameter search took {0:.2f} s".format(t1-t0))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    results = dict(gs_cv=gs.cv_results_)
    # post processing
    # profiling results: mean and standard deviation for testing and training
    mt, st, mtr, sstr = profile_params(gs.cv_results_, scoring, args.classifier)
    results['profile'] = dict(mean_test=mt, std_test=st, mean_train=mtr, std_train=sstr)

    # loop over scoring:
    # get the best param index
    # and for the estimator with these params,
    # calcuate the score.
    results['best_params'] = dict()
    results['score_validation'] = dict()
    results['learning_curve'] = dict()
    results['confusion_matrix'] = dict()
    results['classification_report'] = dict()
    results['bkg_pred'] = dict()

    train_sizes = (np.arange(0.1, 0.9, 0.1) * y_train.shape).astype(np.int)

    for k, v in scoring.items():

        scorer = get_scorer(v)
        # get the best index for parameters
        best_index = np.nonzero(gs.cv_results_[f'rank_test_{k:s}'] == 1)[0][0]
        results['best_params'][k] = copy.deepcopy(default_pars[args.classifier])
        results['best_params'][k].update(gs.cv_results_['params'][best_index])

        # init an estimator with the best parameters
        best_clf = clf[args.classifier](random_state=42, **results['best_params'][k])
        best_clf.fit(X_train, y_train)

        y_pred = best_clf.predict(X_test)

        results['score_validation'][k] = scorer(best_clf, X_test, y_test)

        # create a learning curve for the best classifier
        train_sizes, train_scores, valid_scores = learning_curve(best_clf, X_train, y_train,
                                                                 train_sizes=train_sizes,
                                                                 cv=kf,
                                                                 verbose=1,
                                                                 n_jobs=args.n_jobs)

        results['learning_curve'][k] = (train_sizes, train_scores, valid_scores)

        # get the confusion matrix for the best classifier
        results['confusion_matrix'][k] = confusion_matrix(y_test, y_pred)

        # get the classification report for the best classifier
        results['classification_report'][k] = classification_report(y_test, y_pred,
                                                                    output_dict=True,
                                                                    labels=[0, 1],
                                                                    target_names=['bkg', 'light']
                                                                    )

        fp_test = results['confusion_matrix'][k][0,1]  # false positive
        fp_test_rate = fp_test / y_pred.size  # false positive rate
        results['bkg_pred'][k] = fp_test_rate * y.size


    # save results
    np.save(os.path.join(args.outdir, f'results_ml_{suffix:s}.npy'), results)
