from __future__ import absolute_import, division, print_function
from tensorflow import keras
from ..ml import significance_scorer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def plot_metric(history, ax=None, metric="loss", **kwargs):
    """
    Plot the evolution of a classification metric
    with epocks

    Parameters
    ----------
    history: keras history object
        the classification history

    ax: matplotlib axes object
        axes for plotting

    metric: string
        name of metric to plot

    kwargs: dict
    additional kwargs passed to plot

    Returns
    -------
    matplotlib axes object
    """
    if ax is None:
        ax = plt.gca()

    label = kwargs.pop('label', '')
    ax.semilogy(history.epoch, history.history[metric], label='Train ' + label, **kwargs)

    kwargs.pop('ls', None)
    ax.semilogy(history.epoch, history.history[f'val_{metric}'], label='Val ' + label, ls='--', **kwargs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    return ax

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
    bkg_rate = fp.sum() / y_true.size * N_tot / t_obs
    eff = tp.sum() / y_true.sum()

    return sig, bkg_rate, eff


def plot_sig_vs_thr(model, X, y_true, t_obs_hours, N_tot, step=0.0001):
    """
    Plot the significance as function of class label threshold

    Parameters
    ----------
    model: keras model
        The trained model
    X: array-like
        the test data
    y_true: array-like
        true class labels for data X
    t_obs_hours: float
        total observation time in hours
    N_tot: int
        total number of triggers recorded during t_obs-hours
    step: float
        step size for threshold

    Returns
    -------
    tuple with matplotlib axes object, threshold array,
    significance array, bkg rate array, and efficiency array
    """
    y_pred = model.predict(X)

    threshold = np.arange(step, 1., step)
    significance = np.zeros_like(threshold)
    bkg_rate = np.zeros_like(threshold)
    eff = np.zeros_like(threshold)

    for i, thr in enumerate(threshold):
        significance[i], bkg_rate[i], eff[i] = get_sig_bkg_rate_eff(
            y_true,
            y_pred,
            N_tot,
            t_obs_hours * 3600.,
            thr=thr)

    imax = np.argmax(significance)
    print(f"Max significance: {significance[imax]:.2f} for threshold {threshold[imax]:.4f}"
          f" background rate {bkg_rate[imax]:.2e} and analysis efficiency {eff[imax]:.2f}")

    ax = plt.subplot(311)
    ax.plot(threshold, significance)
    ax.set_ylabel("Significance ($\sigma$)")
    ax.tick_params(labelbottom=False, direction="in")
    ax.grid()
    ax.axvline(threshold[imax], color='k', ls='--', zorder=-1)

    ax = plt.subplot(312)
    ax.plot(threshold, bkg_rate)
    ax.set_yscale("log")
    ax.set_ylabel("Bkg rate (Hz)")
    ax.tick_params(labelbottom=False, direction="in")
    ax.grid()
    ax.set_ylim(5e-6, ax.get_ylim()[1])
    ax.axvline(threshold[imax], color='k', ls='--', zorder=-1)

    ax = plt.subplot(313)
    ax.plot(threshold, eff)
    # ax.set_yscale("log")
    ax.set_ylabel("Efficiency")
    ax.tick_params(direction="in")
    ax.grid()
    ax.axvline(threshold[imax], color='k', ls='--', zorder=-1)

    return ax, threshold, significance, bkg_rate, eff


class SplitData(object):
    """
    Class for data splitting for K-fold cross validation
    and hyper-parameter optimization
    """
    def __init__(self, X, y, n_splits=5, stratify=True, random_state=None):
        self._X = X
        self._y = y
        self._random_state = random_state
        self._kf = None
        self._stratify = stratify
        self._n_splits = n_splits

        self._X_train = None
        self._X_val = None
        self._X_test = None

        self._y_train = None
        self._y_val = None
        self._y_test = None

        self._idx_train = None
        self._idx_val = None
        self._idx_test = None

        self.init_kfold()

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def X_val(self):
        return self._X_val

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def y_val(self):
        return self._y_val

    @property
    def idx_train(self):
        return self._idx_train

    @property
    def idx_test(self):
        return self._idx_test

    @property
    def idx_val(self):
        return self._idx_val

    def init_kfold(self):

        if self._stratify:
            # retain same percentage of bkg/light samples in each fold
            self._kf = StratifiedKFold(n_splits=self._n_splits,
                                       shuffle=True,
                                       random_state=self._random_state)
        else:
            self._kf = KFold(n_splits=self._n_splits,
                             shuffle=True,
                             random_state=self._random_state)

    def get_split(self, n):
        """
        Split the data into training and validation set
        and get the n-th split

        Parameters
        ----------
        n: int
        get the n-th data split

        Returns
        -------
        tuple with X_train, X_test, y_train, y_test
        """

        if n >= self._n_splits:
            raise ValueError("n larger than number of splits, select between 0 and f{self._n_splits - 1:n}")

        for i, (idx_train, idx_test) in enumerate(self._kf.split(self._X, self._y)):
            if i == n:
                self._idx_train = idx_train
                self._idx_test = idx_test
                self._idx_val = idx_test  # validation and test set the same in this case

                self._X_train = self._X[idx_train]
                self._X_test = self._X[idx_test]
                self._X_val = self._X[idx_test]

                self._y_train = self._y[idx_train]
                self._y_test = self._y[idx_test]
                self._y_val = self._y[idx_test]

        return self._X_train, self._X_test, self._y_train, self._y_test

    def get_split_with_test_set(self, n, m):
        """
        Perform 2 splits: set aside test data and then
        split the data into training and validation set
        and get the m-th split for training and validation set
        as well as the n-th split for the final test set

        Parameters
        ----------
        n: int
            get the n-th data split

        m: int
            get the n-th data split

        Returns
        -------
        tuple with X_train, X_val, y_train, y_val
        """

        if n >= self._n_splits or m >= self._n_splits:
            raise ValueError("n or m larger than number of splits, select between 0 and f{self._n_splits - 1:n}")

        for i, (idx_train, idx_test) in enumerate(self._kf.split(self._X, self._y)):
            # set aside test set
            if i == n:
                self._idx_test = idx_test
                self._X_test = self._X[idx_test]
                self._y_test = self._y[idx_test]

            else:
                continue

            # from remaining data, generate training and validation set
            for j, (idx_train_train, idx_val) in enumerate(self._kf.split(self._X[idx_train], self._y[idx_train])):
                if j == m:
                    self._idx_train = idx_train_train
                    self._idx_val = idx_val

                    self._X_train = self._X[idx_train][idx_train_train]
                    self._X_val = self._X[idx_train][idx_val]

                    self._y_train = self._y[idx_train][idx_train_train]
                    self._y_val = self._y[idx_train][idx_val]

        return self._X_train, self._X_val, self._y_train, self._y_val
