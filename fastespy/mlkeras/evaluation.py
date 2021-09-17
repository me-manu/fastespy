from __future__ import absolute_import, division, print_function
from tensorflow import keras
import tensorflow as tf
from .models import train_model
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from sklearn.model_selection import KFold, StratifiedKFold

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

    ax.set_xlabel("Threshold")

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


class SignificanceMetric(keras.metrics.Metric):
    """Class to evaluate significance of signal in LSW experiment"""

    def __init__(self, name='significance',
                 N_tot=1000.,
                 e_d=0.5,
                 n_s=2.8e-5,
                 t_obs=20. * 24. * 3600.,
                 thr=0.5,
                 **kwargs):
        super(SignificanceMetric, self).__init__(name=name, **kwargs)

        self._N_tot = tf.cast(N_tot, self.dtype)
        self._e_d = tf.cast(e_d, self.dtype)
        self._n_s = tf.cast(n_s, self.dtype)
        self._t_obs = tf.cast(t_obs, self.dtype)
        self._thr = thr

        self.significance = self.add_weight(name='significance', initializer='zeros')
        self._n_samples = tf.Variable(0., name='current_sample_size', dtype=self.dtype)
        self._n_true_samples = tf.Variable(0., name='current_number_true_samples', dtype=self.dtype)
        self._tp_tot = tf.Variable(0., name='tp_total', dtype=self.dtype)
        self._fp_tot = tf.Variable(0., name='fp_total', dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.greater_equal(y_pred, 0.5)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        batch_size = tf.cast(tf.shape(y_true)[0], self.dtype)
        self._n_samples.assign_add(batch_size)
        self._n_true_samples.assign_add(tf.reduce_sum(tf.cast(y_true, self.dtype)))

        # calculate efficiency, i.e., the number of true positives over all positives
        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.reduce_sum(tf.cast(tp, self.dtype))
        self._tp_tot.assign_add(tp)
        efficiency = self._tp_tot / self._n_true_samples

        # false positive rate, i.e., false positives over sample size
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fp = tf.reduce_sum(tf.cast(fp, self.dtype))
        self._fp_tot.assign_add(fp)
        fp_rate = self._fp_tot / self._n_samples

        # bkg rate: false positive rate times total number of triggers over obs time
        n_b = fp_rate * self._N_tot / self._t_obs

        # significance
        sig = 2. * (tf.sqrt(n_b + self._e_d * efficiency * self._n_s) - tf.sqrt(n_b)) * tf.sqrt(self._t_obs)

        self.significance.assign(sig)

    def result(self):
        return self.significance

    def reset_states(self):
        # reset all values after each epoch
        self.significance.assign(0.)
        self._fp_tot.assign(0.)
        self._tp_tot.assign(0.)
        self._n_samples.assign(0.)
        self._n_true_samples.assign(0.)


def learning_curve(model, X, y,
                   sample_splits=[0.1, 0.3, 0.5, 0.7, 1.],
                   iter_per_split=5,
                   restore_weights=True,
                   epochs=20,
                   normalizer=None,
                   batch_size=500,
                   random_state=None,
                   stratify=True):
    """
    Compute the learning curve

    Parameters
    ----------
    model: keras model
        The model to compute the learning curve for
    X: array-like
        the data (should be shuffled!)
    y: array-like
        class labels
    sample_splits: array-like
        fraction of data to use to compute learning curve
    iter_per_splits: int
        number of iterations for each fraction of the data
        Data will be split using Kfolds with Stratification
    restore_weights: bool
        restore weights after each iteration
    epochs=20
        Number of epochs the model will be trained for
    normalizer: keras normalizer or None
        normalizer to normalize data

    Returns
    -------
    """
    # save weights
    weights = os.path.join(tempfile.mkdtemp(), f'weights')
    model.save_weights(weights)

    sample_sizes = []
    results_test = {}
    results_train = {}

    for i_frac, fraction in enumerate(sample_splits):
        sample_size = int(y.size * fraction)
        sample_sizes.append(sample_size)

        sd = SplitData(X[:sample_size], y[:sample_size],
                       n_splits=iter_per_split,
                       stratify=stratify,
                       random_state=random_state)

        for i in range(iter_per_split):

            # restore initial weights
            if restore_weights:
                model.load_weights(weights)

            X_train, X_test, y_train, y_test = sd.get_split(i)

            if normalizer is not None:
                normalizer.adapt(X_train)
                X_train = normalizer(X_train)
                X_test = normalizer(X_test)

            history = train_model(model, X_train, y_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  X_val=X_test,
                                  y_val=y_test)

            result_train = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0, return_dict=True)
            result_test = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
            # save results
            if not len(list(result_train.keys())) == len(list(results_train.keys())):
                # first step in loop, set up dicts:
                for k, v in result_train.items():
                    results_train[k] = np.zeros((len(sample_splits), iter_per_split))
                    results_train[k][i_frac, i] = v

                    results_test[k] = np.zeros((len(sample_splits), iter_per_split))
                    results_test[k][i_frac, i] = result_test[k]
            else:
                for k, v in result_train.items():
                    results_train[k][i_frac, i] = v
                    results_test[k][i_frac, i] = result_test[k]

    # restore initial weights
    if restore_weights:
        model.load_weights(weights)

    return sample_sizes, results_train, results_test