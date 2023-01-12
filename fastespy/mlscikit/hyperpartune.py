from __future__ import absolute_import, division, print_function
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import logging
from .feldman_cousins import poissonian_feldman_cousins_interval
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import get_scorer, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import PosixPath

# --- Some dictionaries with standard classifiers --- #
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
        'hidden_layer_sizes': ((100, 100, 100, 100, 100,), (100, 100, 100, 100, 100, 100),
                                (100, 100, 100, 100, 100, 100, 100)),
        'alpha': 10.**np.arange(-4, 0.1, 0.25)
    }
)

# smaller grid
param_grid_coarse = dict(
    dt={'ccp_alpha': np.linspace(0., 0.001, 5),
        'min_samples_split': np.arange(2, 53, 10),
        'max_depth': np.arange(2, 21, 1)
        },
    bdt={'n_estimators': np.arange(100, 1100, 400),
         'learning_rate': np.arange(0.1, 1.1, 0.4),
         'max_depth': np.arange(2, 10, 3)
         },

    rf={'n_estimators': np.arange(100, 600, 200),
        'max_features': np.arange(1, 6, 2),
        'min_samples_split': np.arange(2, 82, 20),
        },
    mlp={
        'hidden_layer_sizes': ((100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100,)),
        #'hidden_layer_sizes': ((30, 30), (30, 30, 30, 30), (100, 100, 100, 100, 100,)),
        'alpha': 10.**np.arange(-4, 0.1, 0.5)
        }
)

# grid for class weights
weights = np.linspace(0.0, 0.99, 100)
#class_weights = [None, "balanced"] + [{0: x, 1: 1. - x} for x in weights]
class_weights = [{0: x, 1: 1. - x} for x in weights]

param_grid_class_weights = dict(
    dt={'ccp_alpha': [0.],
        'min_samples_split': [2],
        'max_depth': [5],
        'class_weight': class_weights
        },
    bdt={'n_estimators': [500.],
         'learning_rate': [0.1],
         'max_depth': [2],
         'class_weight': class_weights
         },

    rf={'n_estimators': [300],
        'max_features': [1],
        'min_samples_split': [2],
        'class_weight': class_weights
        },
    mlp={
        'hidden_layer_sizes': ((100, 100, 100, 100)),
        'alpha': [1e-3],
        'class_weight': class_weights
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


def get_tp_fp_fn(y_true, y_pred, thr=0.5):
    """
    Get the numbers for true positive, false positive, and false negative
    for binary classification for a certain threshold to classify an event as
    a positive sample

    Parameters
    ----------
    y_true: array-like
        True class labels [0, 1]
    y_pred: array-like
        predicted class labels, i.e., real numbers in the interval [0,1]
    thr: float
        Threshold for classification as a positive event (Default: 0.5)

    Returns
    -------
    Tuple with predicted class labels, true positives, false positives, and false negatives
    """
    # check if y_pred has shape with two labels or one label,
    # we need the latter
    if len(y_pred.shape) > 1:
        if y_pred.shape[1] > 2:  # more than two classes trained, can't deal with this
            raise ValueError("More than two classes trained, not implemented")
        elif y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]  # this should be the prob to belonging to light class
        elif y_pred.shape[1] == 1:
            y_pred = np.squeeze(y_pred)

    class_pred = (y_pred > thr).flatten().astype(int)
    tp = (class_pred == 1) & (y_true == 1)
    fp = (class_pred == 1) & (y_true == 0)
    fn = (class_pred == 0) & (y_true == 1)
    return class_pred, tp, fp, fn

def get_sig_bkg_rate_eff(y_true, y_pred, N_tot, t_obs, thr=0.5):
    """
    Compute the significance, background rate, detection efficiency

    Parameters
    ----------
    y_true: array-like
        True class labels [0, 1]
    y_pred: array-like
        predicted class labels, i.e., real numbers in the interval [0,1]
    N_tot: int
        total number of triggers in test and training sample
    t_obs: float
        obervation time in seconds during which N_tot triggers where observed
    thr: float
        Threshold for classification as a positive event (Default: 0.5)

    Returns
    -------
    Tuple with significance, background rate, detection efficiency
    """
    class_pred, tp, fp, fn = get_tp_fp_fn(y_true, y_pred, thr=thr)

    sig = significance_scorer(y_true, class_pred, t_obs=t_obs, N_tot=N_tot)
    #bkg_rate = fp.sum() / y_true.size * N_tot / t_obs
    bkg_rate = fp.sum() / np.sum(~y_true.astype(bool)) * N_tot / t_obs
    eff = tp.sum() / y_true.sum()

    return sig, bkg_rate, eff


def significance(n_b, obs_time, n_s = 2.8e-5, e_d=0.5, e_a=1.):
    """Signficance of a signal given some background rate and obs time"""
    N_b = obs_time * n_b
    N_s = obs_time * n_s
    eps = e_d * e_a
    S = 2. * (np.sqrt(eps * N_s + N_b) - np.sqrt(N_b))
    return S


def significance_scorer(y, y_pred,
                    n_s=2.8e-5,
                    e_d=0.5,
                    t_obs=20. * 24. * 3600.,
                    N_tot=1000
                    ):
    """
    Scorer that scales upwards for better detection significance,
    see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

    Parameters
    ----------
    y: array-like
        true labels

    y_pred: array-like
        predicted labels

    t_obs: float
        observation time in seconds

    n_s: float
        signal rate from regenerated photons in Hz

    e_d: float
        Detector efficiency

    N_tot: int
        Total number of triggers recorded in t_obs

    Returns
    -------
    Detection significance
    """

    # true positive rate
    # this is also the analysis efficiency
    tp_rate = np.sum(y * y_pred) / float(np.sum(y))

    # misidentified background events
    # false positive rate
    #fp_rate = np.sum((y_pred == 1) & (y == 0)) / float(len(y))
    fp_rate = np.sum((y_pred == 1) & (y == 0)) / np.sum(~y.astype(bool))
    # from this, you get the dark current
    n_b = fp_rate * N_tot / t_obs

    S = 2. * (np.sqrt(e_d * tp_rate * n_s + n_b) - np.sqrt(n_b)) * np.sqrt(t_obs)
    if np.isnan(S):
        raise ValueError("Significance is NaN")
    return S


# --- The class for ML and hyperparameter tuning --- #
class MLHyperParTuning(object):
    """
    Class the wraps many of the common tasks for hyperparameter tuning
    using sklearn
    """
    def __init__(self, X, y,
                 X_test=None,
                 y_test=None,
                 idx_test=None,
                 valid_fraction=0.2,
                 stratify=True,
                 random_state=None,
                 n_splits=5):
        """
        Initialize the class

        Parameters
        ----------
        X: array-like
            Data of dimensions n_samples x n_features

        y: array-like
            Data labels of size n_samples

        X_test: array-like
            Test data of dimensions n_samples x n_features
            if provided, `y_test` and `idx_best` have to be given as well and
            `X` and `y` are interpreted as training data sets.

        y_test: array-like
            Labels of test data of dimensions n_samples
            if provided, `X_test` and `idx_test` have to be given as well and
            `X` and `y` are interpreted as training data sets.

        idx_test: array-like
            Index of test data for entire data set.
            if provided, `X_test` and `y_test` have to be given as well and
            `X` and `y` are interpreted as training data sets.

        valid_fraction: fload
            Fraction of data put aside for validation purposes

        shuffle: bool
            Determine whether split of test and training data should
            preserve fraction of class labels of original data

        scale: bool
            Determine whether data should be scaled to zero mean
            and unit variance for each feature

        random_state: None, int, or random state
            the random state to use

        n_splits: int
            Number of splits used for K-fold cross validation
        """

        if X_test is None or y_test is None or idx_test is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                random_state=random_state,
                                                                stratify=y if stratify else None,
                                                                test_size=valid_fraction
                                                                )
            # save indeces of train and test sample
            # of original array
            self._idx_test = np.zeros(y_test.size, dtype=np.int)
            for i, Xi_test in enumerate(X_test):
                diff = np.sum(Xi_test - X, axis=1)
                self._idx_test[i] = np.where(diff == 0.)[0][0]

        else:
            X_train, y_train = X, y
            self._idx_test = idx_test

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        if stratify:
            # retain same percentage of bkg/light samples in each fold
            self._kf = StratifiedKFold(n_splits=n_splits,
                                       shuffle=True,
                                       random_state=random_state)
        else:
            self._kf = KFold(n_splits=n_splits,
                             shuffle=True,
                             random_state=random_state)
        self._sig_score = None
        self._results = None
        self._grid = None
        self._scoring = None
        self._classifier = None
        self._y_pred_test = None
        self._y_pred_train = None
        self._prob_test = None
        self._prob_train = None
        self._t_obs = None
        self._N_tot = None

    def scale_data(self):
        """preprocess data: zero mean, standard deviation of 1"""
        scaler = StandardScaler().fit(self._X_train)
        self._X_train = scaler.transform(self._X_train)
        self._X_test = scaler.transform(self._X_test)

    @property
    def results(self):
        return self._results

    @property
    def grid(self):
        return self._grid

    @property
    def y_train(self):
        return self._y_train

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_pred_test(self):
        return self._y_pred_test

    @property
    def y_pred_train(self):
        return self._y_pred_train

    @property
    def idx_test(self):
        return self._idx_test

    @property
    def classifier(self):
        return self._classifier

    @property
    def scoring(self):
        return self._scoring

    @property
    def t_obs(self):
        return self._t_obs

    @property
    def N_tot(self):
        return self._N_tot

    @property
    def kf(self):
        return self._kf

    @staticmethod
    def transform_data_log(X, y,
                           feature_names,
                           negative_features=['pulse height', 'pulse integral fit']):
        """
        Transform the data into log space
        """
        m = np.ones(X.shape[0], dtype=np.bool)

        for j, fn in enumerate(feature_names):
            if fn == 'constant':
                X[:, j] += 1.

            if fn in negative_features:
                X[:, j] = np.log10(-X[:, j])
            else:
                X[:, j] = np.log10(X[:, j])
            m &= np.isfinite(X[:, j])

        print("Discarded {0:n} triggers in log transform".format(m.size - np.sum(m)))
        return X[m], y[m]

    def transform_data_pca(self, n_components=None):
        """Perform PCA transformation of data (fitted on training data only)"""
        pca = PCA(n_components=n_components)
        pca.fit(self._X_train)
        self._X_train = pca.transform(self._X_train)
        self._X_test = pca.transform(self._X_test)

    def make_sig_scorer(self, t_obs, N_tot):
        """Make a scorer for the significance given some observation time in seconds"""
        self._t_obs = t_obs
        self._N_tot = N_tot
        self._sig_score = make_scorer(significance_scorer,
                                      greater_is_better=True,
                                      t_obs=t_obs,
                                      #N_tot=self._y_test.size + self._y_train.size
                                      N_tot=N_tot
                                      )

    def perform_grid_search(self, classifier,
                            default_pars,
                            param_grid,
                            refit,
                            scoring=None,
                            class_weight_grid=False,
                            n_jobs=8,
                            verbose=1,
                            random_state=None):
        """
        Perform hyperparameter optimazation through a cross validation grid search.

        Parameters
        ----------
        classifier: sklearn classifier object
            The sklearn classifier

        default_pars: dict
            dictionary with default parameters for the classifier

        param_grid: dict
            dictionary with parameters for the grid search

        refit: str
            Name of scorer used for refitting final classifier

        scoring: str or dict or None
            Scorers for which grid search will be performed. If None, use hard coded scoring dict.

        coarse_grid: bool
            if True and param_grid_user not given, use hard coded coarse grid

        class_weight_grid: bool
            if True and param_grid_user not given, use hard coded grid for class weight optimization
            (takes precedence over coarse grid)

        n_jobs: int
            number of parallel jobs

        verbose: int
            verbosity of output

        random_state: int, None
            random state to use
        """

        if class_weight_grid:
            self._grid = copy.deepcopy(param_grid)
            # append the "balanced" weighting
            self._grid['class_weight'].append({})
            for label in np.unique(self._y_train):
                m = self._y_train == label
                w = self._y_train.size / np.unique(self._y_train).size / self._y_train[m].size
                self._grid['class_weight'][-1][label] = w
        else:
            self._grid = param_grid

        self._classifier = classifier

        if scoring is None:
            self._scoring = {'AUC': 'roc_auc',
                             'Accuracy': 'accuracy',
                              # 'Precision': 'precision',
                              # 'Recall': 'recall',
                              # 'F_1': 'f1',
                              # 'F_2': make_scorer(fbeta_score, beta=2),
                              # 'F_{1/2}': make_scorer(fbeta_score, beta=0.5),
                             }
            # add significance score if defined
            if self._sig_score is not None:
                self._scoring['Significance'] = self._sig_score
        else:
            self._scoring = scoring

        gs = GridSearchCV(self._classifier,
                          param_grid=self._grid,
                          scoring=self._scoring,
                          refit=refit,
                          return_train_score=True,
                          cv=self._kf,
                          verbose=verbose,
                          n_jobs=n_jobs
                          )

        t0 = time.time()
        gs.fit(self._X_train, self._y_train)
        t1 = time.time()
        print("The parameter search took {0:.2f} s".format(t1 - t0))

        self._results = dict(gs_cv=gs.cv_results_)

        print("Profiling over parameters")
        self._profile_params()

        print("Refitting on whole test data set and computing learning curve and confusion matrix")
        self._post_processing(default_pars, n_jobs=n_jobs)

    def _profile_params(self):
        """
        Profile the grid parameters.
        For each value of a parameter, compute the best mean test and train scores and standard deviations
        from profiling, i.e., for each value of a parameter
        set the other grid parameters to the values that maximize the score.

        Sets the 'profile' key in the results dict.
        """
        mean_best_test = {}
        mean_best_train = {}
        std_best_test = {}
        std_best_train = {}

        for score in self._scoring.keys():

            mean_best_test[score] = {}
            mean_best_train[score] = {}
            std_best_test[score] = {}
            std_best_train[score] = {}

            for param, v in self._grid.items():

                mean_best_test[score][param] = np.zeros_like(v).astype(np.float)
                mean_best_train[score][param] = np.zeros_like(v).astype(np.float)
                std_best_test[score][param] = np.zeros_like(v).astype(np.float)
                std_best_train[score][param] = np.zeros_like(v).astype(np.float)

                for i, vi in enumerate(v):
                    # create a mask where flattened param array corresponds to k = vi
                    if param == 'hidden_layer_sizes':
                        m = []
                        for x in self._results['gs_cv']['param_hidden_layer_sizes'].data:
                            m.append(x == vi)
                        m = np.array(m)
                    else:
                        m = self._results['gs_cv'][f'param_{param}'] == vi
                    # get the best value for this vi
                    idmax_test = np.argmax(self._results['gs_cv'][f'mean_test_{score}'][m])
                    idmax_train = np.argmax(self._results['gs_cv'][f'mean_train_{score}'][m])

                    mean_best_test[score][param][i] = self._results['gs_cv'][f'mean_test_{score}'][m][idmax_test]
                    std_best_test[score][param][i] = self._results['gs_cv'][f'std_test_{score}'][m][idmax_test]

                    mean_best_train[score][param][i] = self._results['gs_cv'][f'mean_train_{score}'][m][idmax_train]
                    std_best_train[score][param][i] = self._results['gs_cv'][f'std_train_{score}'][m][idmax_train]

        self._results['profile'] = dict(mean_test=mean_best_test,
                                        std_test=std_best_test,
                                        mean_train=mean_best_train,
                                        std_train=std_best_train)

    def _post_processing(self, default_pars, n_jobs=8, step=0.002):
        """
        Retrain the classifier with best parameter set on whole
        training sample for each scorer, compute the learning curve
        and the confusion matrix.
        """
        self._results['best_params'] = dict()
        self._results['score_validation'] = dict()
        self._results['learning_curve'] = dict()
        self._results['classification_report'] = dict()
        for k in ['test', 'train']:
            self._results['confusion_matrix_{0:s}'.format(k)] = dict()
            self._results['bkg_pred_{0:s}'.format(k)] = dict()
            self._results['tp_efficiency_{0:s}'.format(k)] = dict()
            self._results['score_{0:s}'.format(k)] = dict()
            self._results['thr_sig_bkg_eff_{0:s}'.format(k)] = dict()
        self._y_pred_test = dict()
        self._y_pred_train = dict()
        self._prob_test = dict()
        self._prob_train = dict()

        train_sizes = (np.arange(0.1, 0.9, 0.1) * self._y_train.shape).astype(np.int)
        thresholds = np.arange(0., 1. + step, step)
        self._results['thresholds'] = thresholds

        for k, v in self._scoring.items():
            scorer = get_scorer(v)

            # get the best index for parameters
            best_index = np.nonzero(self._results['gs_cv'][f'rank_test_{k:s}'] == 1)[0][0]
            self._results['best_params'][k] = copy.deepcopy(default_pars)
            self._results['best_params'][k].update(self._results['gs_cv']['params'][best_index])

            # init an estimator with the best parameters
            #best_clf = self._classifier(random_state=42, **self._results['best_params'][k])
            best_clf = self._classifier.set_params(**self._results['best_params'][k])
            best_clf.fit(self._X_train, self._y_train)

            self._y_pred_test[k] = best_clf.predict(self._X_test)
            self._y_pred_train[k] = best_clf.predict(self._X_train)
            self._prob_test[k] = best_clf.predict_proba(self._X_test)
            self._prob_train[k] = best_clf.predict_proba(self._X_train)

            self._results['score_test'][k] = scorer(best_clf, self._X_test, self._y_test)
            self._results['score_train'][k] = scorer(best_clf, self._X_train, self._y_train)

            # compute the dependence of significance, bkg rate and efficiency
            # on the threshold value
            for i in range(2):
                sig, bkg, eff = np.zeros(thresholds.size), np.zeros(thresholds.size), np.zeros(thresholds.size)
                for j, thr_j in enumerate(thresholds):
                    sig[j], bkg[j], eff[j] = get_sig_bkg_rate_eff(self._y_test if i else self._y_train,
                                                                  self._prob_test[k][:, 1] if i else
                                                                      self._prob_train[k][:, 1],
                                                                  self._N_tot,
                                                                  self._t_obs, thr_j)

                if i:
                    self._results['thr_sig_bkg_eff_test'][k] = (sig, bkg, eff)
                else:
                    self._results['thr_sig_bkg_eff_train'][k] = (sig, bkg, eff)

            train_sizes, train_scores, valid_scores = learning_curve(best_clf, self._X_train,
                                                                     self._y_train,
                                                                     train_sizes=train_sizes,
                                                                     cv=self._kf,
                                                                     verbose=1,
                                                                     scoring=scorer,
                                                                     n_jobs=n_jobs)

            self._results['learning_curve'][k] = (train_sizes, train_scores, valid_scores)

            # get the confusion matrix for the best classifier
            self._results['confusion_matrix_test'][k] = confusion_matrix(self._y_test,
                                                                         self._y_pred_test[k])
            self._results['confusion_matrix_train'][k] = confusion_matrix(self._y_train,
                                                                          self._y_pred_train[k])

            # get the classification report for the best classifier
            self._results['classification_report'][k] = classification_report(self._y_test,
                                                                              self._y_pred_test[k],
                                                                              output_dict=True,
                                                                              labels=[0, 1],
                                                                              target_names=['bkg', 'light']
                                                                              )

            # compute the background rate for test and train sample
            self._results['bkg_pred_train'][k], self._results['tp_efficiency_train'][k] = \
                self.compute_bkg_rate_tp_efficiency(self._results['confusion_matrix_train'][k],
                                                    y=self._y_train,
                                                    #n_triggers=self._y_test.size + self._y_train.size
                                                    n_triggers=self._N_tot
                                                    )

            self._results['bkg_pred_test'][k], self._results['tp_efficiency_test'][k] = \
                self.compute_bkg_rate_tp_efficiency(self._results['confusion_matrix_test'][k],
                                                    y=self._y_test,
                                                    #n_triggers=self._y_test.size + self._y_train.size
                                                    n_triggers=self._N_tot
                                                    )

    @staticmethod
    def compute_bkg_rate_tp_efficiency(confusion_matrix, y, n_triggers):
        # compute the background rate for test and train sample
        fp = confusion_matrix[0, 1]  # false positive
        #fp_rate = fp / y.size  # false positive rate
        fp_rate = fp / (y == 0).sum()  # false positive rate
        bkg_rate = fp_rate * n_triggers

        # efficiency of identifying light
        tp = confusion_matrix[1, 1]  # true positive
        tp_efficiency = tp / (y == 1).sum()

        return bkg_rate, tp_efficiency

    @staticmethod
    def plot_confusion_matrix(results, scoring, classifier="", path=PosixPath("../")):
        """
        Plot the confusion matrix for each scorer
        """
        if path is not None and not path.exists():
            path.mkdir(parents=True)

        if type(scoring) is list:
            scorer_names = scoring
        elif type(scoring) is dict:
            scorer_names = scoring.keys()
        else:
            raise ValueError("type of scoring not understood")

        for score in scorer_names:
            disp = ConfusionMatrixDisplay(np.array(results['confusion_matrix_test'][score]),
                                          display_labels=['bkg', 'light'])
            disp.plot(cmap=plt.cm.Blues,
                      values_format="d")
            plt.title(f"{classifier}" + f" {score} ")
            if path is not None:
                plt.savefig(path / f"confusion_matrix_{score}_{classifier}.png", dpi=150)
        if path is not None:
            plt.close("all")

    @staticmethod
    def plot_learning_curve(results, classifier="", select_score=None, path=PosixPath("../")):
        """
        Plot the learning curve

        Parameters
        ----------
        select_score: str
            name of score to compute learning curve for. If None, compute for all used scorers
        """
        if not path is None and not path.exists():
            path.mkdir(parents=True)

        for i, (score, val) in enumerate(results['learning_curve'].items()):
            if select_score is not None:
                if not score == select_score:
                    continue
            plt.figure()

            train_sizes, train_scores, valid_scores = val

            plt.plot(train_sizes, train_scores.mean(axis=1),
                     marker='o',
                     label=score + " Train",
                     ls='--',
                     color=plt.cm.tab10(0.),
                     )

            plt.fill_between(train_sizes,
                             train_scores.mean(axis=1) - np.sqrt(train_scores.var()),
                             y2=train_scores.mean(axis=1) + np.sqrt(train_scores.var()),
                             alpha=0.3,
                             color=plt.cm.tab10(0.),
                             zorder=-1
                             )

            plt.plot(train_sizes, valid_scores.mean(axis=1),
                     marker='o',
                     label=score + " valid", ls='-',
                     color=plt.cm.tab10(0.1),
                     )
            plt.fill_between(train_sizes,
                             valid_scores.mean(axis=1) - np.sqrt(valid_scores.var()),
                             y2=valid_scores.mean(axis=1) + np.sqrt(valid_scores.var()),
                             alpha=0.3,
                             color=plt.cm.tab10(0.1),
                             zorder=-1
                             )

            plt.legend(title=classifier)
            plt.grid()
            plt.xlabel("Sample Size")
            plt.ylabel("Score")
            if path is not None:
                plt.savefig(path / f"learning_curve_{score}_{classifier}.png", dpi=150)

        if path is not None:
            plt.close("all")

    @staticmethod
    def plot_parameter_profiles(results, scoring, classifier, path=PosixPath("../")):
        """Plot the parameter profiles for each scorer"""

        if path is not None:
            if not path.exists():
                path.mkdir(parents=True)

        for i, score in enumerate(scoring):
            n_par = len(results['profile']['mean_test'][score].keys())
            plt.figure(figsize=(4 * n_par, 4))
            ax_list = []
            for j, par in enumerate(results['profile']['mean_test'][score].keys()):

                ax = plt.subplot(1, len(results['profile']['mean_test'][score].keys()), j + 1)

                if par == 'hidden_layer_sizes':

                    x = np.unique([np.sum(results['gs_cv'][f'param_{par}'].data[i]) for i in
                                   range(results['gs_cv'][f'param_{par}'].data.size)])

                elif par == 'class_weight':
                    x = np.array([x[0] for x in results['gs_cv'][f'param_{par}'].data])
                else:
                    x = np.unique(results['gs_cv'][f'param_{par}'].data).astype(np.float)

                for i_t, t in enumerate(['train', 'test']):
                    ax.plot(x, results['profile'][f'mean_{t:s}'][score][par],
                            color=plt.cm.tab10(0.1 * i_t),
                            ls='-' if t == 'test' else '--',
                            label=t
                            )

                    ax.fill_between(x, results['profile'][f'mean_{t:s}'][score][par] -
                                           0.5 * results['profile'][f'std_{t:s}'][score][par],
                                    y2=results['profile'][f'mean_{t:s}'][score][par] +
                                           0.5 * results['profile'][f'std_{t:s}'][score][par],
                                    color=plt.cm.tab10(0.1 * i_t),
                                    alpha=0.3)

                if not j:
                    ax.set_ylabel(score)
                    ax.legend()
                    vy = list(ax.get_ylim())
                else:
                    ax.tick_params(labelleft=False)
                    v_now = ax.get_ylim()
                    vy[0] = v_now[0] if v_now[0] < vy[0] else vy[0]
                    vy[1] = v_now[1] if v_now[1] > vy[1] else vy[1]

                ax_list.append(ax)
                ax.grid()
                ax.set_xlabel(par)

                if par == 'alpha' and classifier == 'mlp':
                    ax.set_xscale('log')

                for a in ax_list:
                    a.set_ylim(vy)

            plt.subplots_adjust(wspace=0.1)
            if path is not None:
                plt.savefig(path / f"parameter_profiles_{score}_{classifier}.png", dpi=150)

        if path is not None:
            plt.close("all")

    @staticmethod
    def plot_misidentified_time_lines(result,
                                      scorer,
                                      data_time,
                                      data_voltage,
                                      X=None,
                                      classifier="",
                                      feature_names=None,
                                      path=PosixPath("../"),
                                      plot_false_positive=True,
                                      save_plot=True
                                      ):
        """
        Plot the misidentified background time lines
        """
        if plot_false_positive:
            # plot false signal time lines
            m_misid = (result['y_test'] == 0) & (result['y_pred_test'][scorer] == 1)
        else:
            # else plot false background time lines
            m_misid = (result['y_test'] == 1) & (result['y_pred_test'][scorer] == 0)

        n_misid = np.sum(m_misid)

        if not n_misid:
            print("No misidentified pulses found, returning.")
            return

        plt.figure(figsize=(6 * 4, n_misid))
        iplot = 1
        for i, idx in enumerate(result['idx_test'][m_misid]):
            ax = plt.subplot(int(n_misid / 4) + 1, 4, iplot)

            if X is not None and feature_names is not None:
                label = ""
                for j, fn in enumerate(feature_names):
                    label += "{0:s}: {1:.2e}".format(fn, X[idx, j])
                    if not j == len(feature_names) - 1:
                        label += "\n"
            else:
                label = ""
            ax.plot((data_time[idx] - data_time[idx, 0]) * 1e6, data_voltage[idx], lw=1, label=label)

            title_string = "ID {1:d} Sig. Prob. = {0:.3f}".format(result['prob_test'][scorer][m_misid][i, 1], idx)
            ax.legend(fontsize='small', title=title_string, loc='lower right')
            iplot += 1

        plt.suptitle(f"{classifier:s} {scorer:s}", fontsize='xx-large')
        plt.subplots_adjust(top=0.95)

        if not path.exists():
            path.mkdir(parents=True)

        if save_plot:
            if plot_false_positive:
                plt.savefig(path / f"misid_events_fp_{scorer:s}_{classifier}.png", dpi=150)
            else:
                plt.savefig(path / f"misid_events_fn_{scorer:s}_{classifier}.png", dpi=150)
            plt.close("all")

    @staticmethod
    def print_performance_report(results, scoring, t_tot_s):
        keys = ['bkg_pred', 'tp_efficiency', 'score']

        for k in scoring.keys():
            for ki in keys:
                print("==== {0:s} : {1:s} ====".format(k, ki))
                for t in ['train', 'test']:
                    if not ki == 'score':
                        if ki == 'bkg_pred':
                            c = t_tot_s
                        else:
                            c = 1.
                        print("{0:s}: {1:.3e}".format(t,
                                                      results['{0:s}_{1:s}'.format(ki, t)][k]
                                                      / c))
                    else:
                        print("{0:s}: {1:.3e}".format(t,
                                                      results['{0:s}_{1:s}'.format(ki, t)][k]))

            print("==== {0:s} : {1:s} ====".format(k, "Significance"))
            for t in ['train', 'test']:
                t_tot = t_tot_s
                sig = significance(n_b=results['bkg_pred_{0:s}'.format(t)][k] / t_tot,
                                   obs_time=t_tot,
                                   e_a=results['tp_efficiency_{0:s}'.format(t)][k]
                                   )
                print("{0:s}: {1:.3f}".format(t, sig))

    def make_result_dict(self):
        result = {}
        result['results'] = copy.deepcopy(self._results)
        result['y_pred_test'] = self._y_pred_test
        result['y_pred_train'] = self._y_pred_train
        result['prob_test'] = self._prob_test
        result['prob_train'] = self._prob_train
        result['X_train'] = self._X_train
        result['X_test'] = self._X_test
        result['y_train'] = self._y_train
        result['y_test'] = self._y_test
        result['idx_test'] = self._idx_test
        result['scoring'] = list(self._scoring.keys())
        result['grid'] = self._grid
        result['t_obs'] = self._t_obs
        return result


def run_hyper_par_opt(X, y,
                      idx_train,
                      idx_test,
                      feature_names,
                      classifier,
                      param_grid,
                      default_pars={},
                      t_tot_hrs=500.,
                      N_bkg_trigger=1000,
                      data=None,
                      kfolds=5,
                      classifier_name="clf",
                      random_state=42,
                      use_pca=False,
                      out_path=PosixPath("../"),
                      n_jobs=8):
    """
    Run the hyper-parameter tuning for one split into validation and training data by splitting
    the training data again with K-fold cross validation.

    Parameters
    ----------
    X: array-like
        The input data in format [n_samples, n_features]
    y: array-like
        The class labels in format [n_samples]
    idx_train: array-like
        list of indeces for samples from X to use for training
    idx_test: array-like
        list of indeces for samples from X to use for validation
    feature_names: array of strings
        Names of features
    classifier: sklearn classifier object
        The sklearn classifier
    param_grid: dict
        dictionary with parameters for the grid search
    default_pars: dict
        dictionary with default parameters for the classifier
    t_tot_hrs: float
        total number of hours of data taking
    N_bkg_trigger: int
        total number of background triggers taken during t_tot_hrs
    data: dict
        dict with raw data (to plot the misidentified time lines)
    kfolds: int
        number of K folds
    classifier_name: str
        name of classifier
    out_path: PosixPath object
        The output dir
    random_state: None, int, or random state
        the random state / seed to use
    use_pca: bool
        if true, apply a PCA transform
    n_jobs: int
        number of cores to use
    """

    ml_tune = MLHyperParTuning(X[idx_train], y[idx_train],
                               X_test=X[idx_test],
                               y_test=y[idx_test],
                               idx_test=idx_test,
                               valid_fraction=1. / kfolds,
                               stratify=True,
                               random_state=random_state,
                               n_splits=kfolds)
    ml_tune.scale_data()
    if use_pca:
        logging.info("Transforming data using PCA")
        ml_tune.transform_data_pca()

    ml_tune.make_sig_scorer(t_obs=t_tot_hrs * 3600., N_tot=N_bkg_trigger)
    ml_tune.perform_grid_search(classifier=classifier,
                                default_pars=default_pars,
                                param_grid=param_grid,
                                refit='Significance',
                                n_jobs=n_jobs)
    # output plots
    ml_tune.plot_confusion_matrix(ml_tune.results, ml_tune.scoring, path=out_path, classifier=classifier_name)
    ml_tune.plot_learning_curve(ml_tune.results, path=out_path, classifier=classifier_name)
    # generate results dict
    results = ml_tune.make_result_dict()
    if data is not None:
        ml_tune.plot_misidentified_time_lines(results,
                                              scorer="Significance",
                                              data_time=data['time'],
                                              data_voltage=data['data'],
                                              X=X,
                                              feature_names=feature_names,
                                              classifier=classifier_name,
                                              path=out_path
                                              )
    # output performance
    logging.info("Printing performance:")
    ml_tune.print_performance_report(ml_tune.results, ml_tune.scoring, ml_tune.t_obs)
    # Add Feldman & Cousins confidence interval for dark current rate
    logging.info("Running Feldman & Cousins confidence interval estimation for dark current")
    n_b = 0
    n_obs = np.arange(ml_tune.results['confusion_matrix_test']['Significance'][0, 1] * 5)
    mus = np.linspace(0, ml_tune.results['confusion_matrix_test']['Significance'][0, 1] * 3, 2401)
    alpha = 0.9
    lower_limits_mu, upper_limits_mu = poissonian_feldman_cousins_interval(
        n_obs=n_obs,
        n_b=n_b,
        mus=mus,
        alpha=alpha,
        fix_discrete_n_pathology=False)
    # n_jobs=args.n_jobs)
    lower_limits = lower_limits_mu[:, 0]
    upper_limits = upper_limits_mu[:, 0]
    # add F&C result to result dict
    results['dark_current'] = {}
    for k in ml_tune.scoring.keys():
        results['dark_current'][k] = np.array([
            lower_limits[ml_tune.results['confusion_matrix_test']['Significance'][0, 1]] * (
                    ml_tune.y_test.size + ml_tune.y_train.size) / ml_tune.y_test.size,
            ml_tune.results['bkg_pred_test']['Significance'],
            upper_limits[ml_tune.results['confusion_matrix_test']['Significance'][0, 1]] * (
                    ml_tune.y_test.size + ml_tune.y_train.size) / ml_tune.y_test.size
        ]) / ml_tune.t_obs
        logging.info("dark current {0:s}: {1}".format(k, results['dark_current'][k]))
    return results


def run_hyper_par_opt_all_folds(X, y,
                                feature_names,
                                classifier,
                                param_grid,
                                default_pars={},
                                t_tot_hrs=500.,
                                N_bkg_trigger=1000,
                                data=None,
                                kfolds=5,
                                classifier_name="clf",
                                out_dir=PosixPath("../"),
                                random_state=42,
                                log_data=False,
                                use_pca=False,
                                n_jobs=8):
    """
    Run the hyper-parameter tuning over all k-folds and generate analysis plots

    Parameters
    ----------
    X: array-like
        The input data in format [n_samples, n_features]
    y: array-like
        The class labels in format [n_samples]
    feature_names: array of strings
        Names of features
    classifier: sklearn classifier object
        The sklearn classifier
    default_pars: dict
        dictionary with default parameters for the classifier
    param_grid: dict
        dictionary with parameters for the grid search
    t_tot_hrs: float
        total number of hours of data taking
    N_bkg_trigger: int
        total number of background triggers taken during t_tot_hrs
    data: dict
        dict with raw data (to plot the misidentified time lines)
    kfolds: int
        number of K folds
    classifier_name: str
        name of classifier
    out_dir: PosixPath object
        The output dir
    random_state: None, int, or random state
        the random state / seed to use
    log_data: bool
        if true, log transform the data
    use_pca: bool
        if true, apply a PCA transform
    n_jobs: int
        number of cores to use
    """

    if log_data:
        logging.info("Transforming data to log space")
        X, y = MLHyperParTuning.transform_data_log(X, y, feature_names)

    # perform stratified K fold
    skf = StratifiedKFold(n_splits=kfolds, random_state=random_state, shuffle=True)
    skf.get_n_splits(X, y)
    # perform the hyper parameter optimization
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        logging.info("Running optimization for split {0:n} / {1:n}".format(i + 1, kfolds))
        out_path = out_dir / "{0:05n}".format(i + 1)
        if not out_path.exists():
            out_path.mkdir(parents=True)

        results = run_hyper_par_opt(X, y,
                                    idx_train=train_idx,
                                    idx_test=test_idx,
                                    feature_names=feature_names,
                                    classifier=classifier,
                                    param_grid=param_grid,
                                    default_pars=default_pars,
                                    t_tot_hrs=t_tot_hrs,
                                    N_bkg_trigger=N_bkg_trigger,
                                    data=data,
                                    kfolds=kfolds,
                                    classifier_name=classifier_name,
                                    random_state=random_state,
                                    use_pca=use_pca,
                                    out_path=out_path,
                                    n_jobs=n_jobs)

        np.save(out_path / f"r{classifier_name}_cleaned_reduced.npy", results)
