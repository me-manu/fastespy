import numpy as np
import argparse
import logging
import time
import os
import random
from collections.abc import Iterable

from pathlib import PosixPath
from fastespy.utils import init_logging
from fastespy.io.readpydata import load_data_rikhav
from fastespy.mlkeras.evaluation import get_sig_vs_thr_keras, calc_cam_binary_classification_utl
from fastespy.timeline.processing import sample_derivative
import matplotlib.pyplot as plt

from fastespy.mlkeras import models
from fastespy.plotting import plot_metric
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from fastespy.mlkeras.models import time_series_z_normalization

from fastespy.mlkeras.evaluation import (
     SplitData,
     SignificanceMetric
)
from fastespy.plotting import (
    plot_performance_vs_threshold,
    plot_cam_timeline,
    plot_misided_timelines
)

from scipy import signal

plt.rcParams['axes.labelsize'] = 'x-large'
#plt.rcParams["font.family"] = "serif"
plt.rcParams["font.family"] = "sans"
#plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.serif"] = ["UHHSans"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.dpi"] = 120
for tick in ['xtick', 'ytick']:
    plt.rcParams[f'{tick}.labelsize'] = 'x-large'
    plt.rcParams[f'{tick}.minor.pad'] = 3.9
    plt.rcParams[f'{tick}.minor.size'] = 2.5
    plt.rcParams[f'{tick}.minor.width'] = 1. # 0.6
    plt.rcParams[f'{tick}.major.pad'] = 4.0
    plt.rcParams[f'{tick}.major.size'] = 4.0
    plt.rcParams[f'{tick}.major.width'] = 1.2 # 0.8

# define light cleaning cuts
light_cleaning_cuts = {
    "chi2 reduced": "chi2 < 6.",
    "decay time": "decay < 10.e-6",
    "trigger time": "(trigger >= 29.5e-6) & (trigger <= 30.8e-6)",  # from gaussian fit, 5 sigma interval
}

remove = ['data', 'time', 'pulse integral raw', 'voltage error',
          'error', 'start time in hrs', 'end time in hrs',
          'trigger time'
          ]

def filter_timelines(X, f_sample, f_max=1.e6, norder=3):
    """
    Apply lowpass filter to time line

    Parameters
    ----------
    X: array-like
        time line array with shape n_samples x n_time_steps

    f_sample: float
        Sampling frequency in Hz.

    f_max: float
        Maximum frequeuncy, i.e. frequency where low pass filter gives -3dB attenuation

    norder: int
        order of the low pass filter

    Returns
    -------
    filtered time line with same shape as input time line
    """

    # calculate the Nyquist frequency
    f_Nyq = X.shape[1] / 2. / (X.shape[1] - 1) * f_sample

    # calculate filter
    if norder > 0:
        b, a = signal.butter(norder, f_max / f_Nyq)
        xf = signal.filtfilt(b, a, X)
    else:
        xf = X
    return xf


class MinMaxTimeSeriesScaler(object):
    """
    Class for min max scaling of time series data
    """
    def __init__(self, feature_range=(0.,1.)):
        self._min_scale = feature_range[0]
        self._max_scale = feature_range[1]

    def fit(self, X, axis=None):
        """
        Adapt the Scaler to time series data

        Parameters
        ----------
        X: array_like
            Time series data of shape (n_samples, n_data_points)

        axis: int or None
            axis to normalize over
        """
        self._data_min = X.min(axis=axis)
        self._data_max = X.max(axis=axis)

    def transform(self, X):
        """
        Transform data:
        X_transform = (X - min) / (max - min)
        where min and max where found by fit function

        Parameters
        ----------
        X: array_like
            Time series data of shape (n_samples, n_data_points)

        Returns
        -------
        Transformed data
        """
        X_transform = (X - self._data_min)
        X_transform /= self._data_max - self._data_min
        X_transform *= self._max_scale - self._min_scale
        X_transform += self._min_scale

        return X_transform

def augment_imbalanced_binary_data(X, y, random_state=None, shuffle=False):
    """
    Augment an imbalanced data set through bootstrapping.
    Draw samples with replacement from minority class
    such that you end up with equal number of samples in both classes

    Parameters
    ----------
    X: array-like
        data of the shape (n_samples, n_features)
    y: labels
    random_state: int or None
        random state for seed

    Returns
    -------
    augmented data set X, y
    """
    idx_class0 = np.where(~y)[0]
    idx_class1 = np.where(y)[0]

    np.random.seed(random_state)

    if idx_class0.size > idx_class1.size:
        idx_class1 = np.random.choice(idx_class1, size=idx_class0.size, replace=True)
    elif idx_class0.size < idx_class1.size:
        idx_class0 = np.random.choice(idx_class0, size=idx_class1.size, replace=True)

    # already same size
    else:
        return X, y

    X_new = np.vstack([X[idx_class0], X[idx_class1]])
    y_new = np.concatenate([y[idx_class0], y[idx_class1]])

    if shuffle:
        np.random.shuffle(X_new)
        np.random.shuffle(y_new)

    return X_new, y_new


def build_model_full_fcn(input_shape, metrics, output_bias=None, learning_rate=1e-2):

    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same', name="conv1")(input_layer)
    conv1 = keras.layers.BatchNormalization(name="BN1")(conv1)
    conv1 = keras.layers.Activation(activation='relu', name="relu1")(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', name="conv2")(conv1)
    conv2 = keras.layers.BatchNormalization(name="BN2")(conv2)
    conv2 = keras.layers.Activation('relu', name="relu2")(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same', name="conv3")(conv2)
    conv3 = keras.layers.BatchNormalization(name="BN3")(conv3)
    conv3 = keras.layers.Activation('relu', name="relu3")(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D(name="gap")(conv3)

    output_layer = keras.layers.Dense(1, activation='sigmoid', name='output',
                                      bias_initializer=output_bias)(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )

    return model

def build_model_test0(input_shape, metrics, output_bias=None, learning_rate=1e-2):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv1")(input_layer)
    conv1 = keras.layers.BatchNormalization(name="BN1")(conv1)
    conv1 = keras.layers.Activation(activation='relu', name="relu1")(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv2")(conv1)
    conv2 = keras.layers.BatchNormalization(name="BN2")(conv2)
    conv2 = keras.layers.Activation('relu', name="relu2")(conv2)

    gap_layer = keras.layers.GlobalAveragePooling1D(name="gap")(conv2)

    output_layer = keras.layers.Dense(1, activation='sigmoid', name='output',
                                      bias_initializer=output_bias)(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )

    return model

def build_model_test0b(input_shape, metrics, output_bias=None, learning_rate=1e-2):
    """
    like model test0 but explicitly with 2 classes
    for CAM
    """
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv1")(input_layer)
    conv1 = keras.layers.BatchNormalization(name="BN1")(conv1)
    conv1 = keras.layers.Activation(activation='relu', name="relu1")(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv2")(conv1)
    conv2 = keras.layers.BatchNormalization(name="BN2")(conv2)
    conv2 = keras.layers.Activation('relu', name="relu2")(conv2)

    gap_layer = keras.layers.GlobalAveragePooling1D(name="gap")(conv2)

    output_layer = keras.layers.Dense(2, activation='softmax', name='output',
                                      bias_initializer=output_bias)(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )

    return model

def build_model_test_modular(input_shape,
                             metrics,
                             n_convs=2,
                             bn=True,
                             filters=64,
                             kernel_size=3,
                             strides=1,
                             output_bias=None,
                             learning_rate=1e-2):
    """
    like model test0 but explicitly with 2 classes
    for CAM
    """
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    if not isinstance(kernel_size, Iterable):
        kernel_size = [kernel_size for _ in range(n_convs)]

    if not isinstance(strides, Iterable):
        strides = [strides for _ in range(n_convs)]

    if not isinstance(filters, Iterable):
        filters = [filters for _ in range(n_convs)]

    for i in range(n_convs):
        x = keras.layers.Conv1D(filters=filters[i],
                                kernel_size=kernel_size[i],
                                strides=strides[i],
                                padding='same',
                                name=f"conv{i+1:n}")(x if i else input_layer)
        if bn:
            x = keras.layers.BatchNormalization(name=f"BN{i+1:n}")(x)
        x = keras.layers.Activation(activation='relu', name=f"relu{i+1:n}")(x)

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)

    output_layer = keras.layers.Dense(2, activation='softmax', name='output',
                                      bias_initializer=output_bias)(x)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )

    return model


def build_model_test1(input_shape, metrics, output_bias=None, learning_rate=1e-2):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv1")(input_layer)
    conv1 = keras.layers.BatchNormalization(name="BN1")(conv1)
    conv1 = keras.layers.Activation(activation='relu', name="relu1")(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', name="conv2")(conv1)
    conv2 = keras.layers.BatchNormalization(name="BN2")(conv2)
    conv2 = keras.layers.Activation('relu', name="relu2")(conv2)

    conv3 = keras.layers.Conv1D(64, kernel_size=3, padding='same', name="conv3")(conv2)
    conv3 = keras.layers.BatchNormalization(name="BN3")(conv3)
    conv3 = keras.layers.Activation('relu', name="relu3")(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D(name="gap")(conv3)

    output_layer = keras.layers.Dense(1, activation='sigmoid', name='output',
                                      bias_initializer=output_bias)(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )

    return model

def fit_fcn_model(model, x_train, y_train, x_val, y_val, out_dir, 
                  callbacks=None,
                  batch_size=2048,
                  epochs=50,
                  suffix='',
                  verbose=1):

    if callbacks is None:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                out_dir / "./best_model{0:s}.hdf5".format(suffix), save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
        ]

    start_time = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)

    duration = time.time() - start_time

    model.save(out_dir / 'last_model{0:s}.hdf5'.format(suffix))
    model = keras.models.load_model(out_dir / "best_model{0:s}.hdf5".format(suffix),
                                    custom_objects=dict(SignificanceMetric=SignificanceMetric))

    if verbose:
        print("training took {0:.2f}s".format(duration))

    # save training history
    np.save(out_dir / "history{0:s}.npy".format(suffix), history.history)

    return history


if __name__ == "__main__":
    usage = "usage: %(prog)s -i indir -o outdir "
    description = "Perform a hyper parameter optimization for machine learning on pulse fit results"
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--indir', required=True, help='Path with npy files for all triggers')
    parser.add_argument('-o', '--outdir', required=True, help='Path to output file')
    parser.add_argument('-s', '--suffix', help='Path to output file', default="")
    parser.add_argument('--bkg-type', help='Bkg type', default="intrinsic")
    parser.add_argument('--overwrite', help='retrain and overwirte', action="store_true")
    parser.add_argument('--use-categorical', help='model will use'
                        'categorical cross entropy',
                        action="store_true")
    parser.add_argument('--use-derivative', help='Also use time series derivative',
                        action="store_true")
    parser.add_argument('--random_state', help='Random state', type=int, default=42)
    parser.add_argument('--i-split', help='Which train / test split to use', type=int, default=0)
    parser.add_argument('--low-pass-freq',
                        help='if given, apply low pass filter to data above this frequency',
                        type=float)

    args = parser.parse_args()
    init_logging("INFO", color=True)

    # make sure random seeds are the same
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(args.random_state)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(args.random_state)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    #tf.random.set_seed(args.random_state)
    # for later versions:
    tf.compat.v1.set_random_seed(args.random_state)
    # 5. Configure a new global `tensorflow` session
    # for later versions:
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # get the files
    in_dir = PosixPath(args.indir)
    out_dir = PosixPath(args.outdir)
    out_dir = out_dir.joinpath(f"{args.i_split + 1:05n}")
    print(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    files = list(in_dir.glob("*.npy"))

    logging.info("Using files:")
    for f in files:
        if 'light' in str(f) or args.bkg_type in str(f):
            logging.info(f)

    logging.info("Using cleaning cuts {}".format(light_cleaning_cuts))

    # read the data
    result, data, t_tot_hrs = load_data_rikhav(files,
                                               feature_names=[],
                                               light_cleaning_cuts=light_cleaning_cuts,
                                               bkg_type=args.bkg_type
                                               )

    print(data['time'].shape)
    print(data['data'].shape)  # n samples x n features
    print(result['type'].shape)  # n samples
    print(np.sum(result['type']), np.sum(~result['type']))  # n samples


    # modify data for testing

    # this worked rather well for extrinsics from May 2021
    batch_size = 100
    batch_size = 50
    epochs = 500
    epochs = 250
    factor_x = 1e6  # seconds to micro seconds
    factor_y = 1e3  # V to mV
    #downsample = 10  # use only every n-th data point / decrease sampling frequency by this amount
    #cut_start = int(data['time'].shape[1] * 0.05)
    #cut_end = int(data['time'].shape[1] * 0.4)

    #downsample = 10  # use only every n-th data point / decrease sampling frequency by this amount
    #cut_start = int(data['time'].shape[1] * 0.05)
    #cut_end = int(data['time'].shape[1] * 0.4)

    # second try
    cut_start = int(data['time'].shape[1] * 0.1)# this was ok for extrinsics
    cut_end = int(data['time'].shape[1] * 0.3)  # this was ok for extrinsics

    downsample = 4  # use only every n-th data point / decrease sampling frequency by this amount
    #downsample = 2  # use only every n-th data point / decrease sampling frequency by this amount
    #downsample = 1  # use only every n-th data point / decrease sampling frequency by this amount
    # 1000 samples is hard on my memory with 64 filters. Works with 16

    f_sample = np.unique(1. / np.round(np.diff(data['time'][0,::downsample]), 10))[0]
    print(f"Sampling frequency: {f_sample:.2e} Hz")

    suffix = "{5:s}sampling_freq{0:.1f}MHz_start{1:n}" \
             "_end{2:n}_batch_size{3:n}_epochs{4:n}".format(f_sample / 1e6,
                                                            cut_start,
                                                            cut_end,
                                                            batch_size,
                                                            epochs,
                                                            args.suffix
                                                            )

    # define data
    X_raw = data['data'][:, cut_start:cut_end:downsample] * factor_y
    y_raw = result['type']

    if args.low_pass_freq is not None:
        n_order = 3
        X_raw = filter_timelines(X_raw, f_sample,
                                 f_max=args.low_pass_freq, norder=n_order)
        suffix += "_lpfilter{0:.1f}MHz".format(args.low_pass_freq / 1e6)

    # augment data
    #X, y = augment_imbalanced_binary_data(X_raw, y_raw, random_state=42, shuffle=True)
    X, y = X_raw, y_raw

    # plot some raw data samples ---------------------------- #
    # mean and averages
    plt.figure(1)
    for i in range(10):
        plt.plot(X[i])
    plt.savefig(out_dir / f'init_data_ex_{suffix:s}.png')

    plt.figure(2, figsize=(6,8))
    ax_mean = plt.subplot(211)
    ax_std = plt.subplot(212)
    ax_mean.plot(X[y].mean(axis=0), label="Sig. mean")
    ax_mean.plot(X[~y].mean(axis=0), label="Bkg. mean")
    ax_mean.legend()
    ax_std.plot(X[y].std(axis=0), label="Sig. STD")
    ax_std.plot(X[~y].std(axis=0), label="Bkg. STD")
    ax_std.legend()
    plt.savefig(out_dir / f'mean_std_{suffix:s}.png')
    # ------------------------------------------------------- #

    # split the data
    sd = SplitData(X, y, n_splits=5, stratify=True, random_state=42)

    # get one split, set aside some testing data
    X_train, X_val, y_train, y_val = sd.get_split_with_test_set(m=args.i_split, n=0)

    # z normalization for time series
    X_train = time_series_z_normalization(X_train, axis=1)
    X_val = time_series_z_normalization(X_val, axis=1)

    #X_train = time_series_z_normalization(X_train, axis=1, invert=True)
    #X_val = time_series_z_normalization(X_val, axis=1, invert=True)

    # scale the time series
    #scaler = MinMaxTimeSeriesScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_val = scaler.transform(X_val)

    # plot some transformed data samples
    plt.figure(3)
    for i in range(10):
        plt.plot(X_train[~y_train][i])
    plt.savefig(out_dir / f'norm_bkg_data_{suffix:s}.png')

    # plot some transformed data samples
    plt.figure(4)
    for i in range(10):
        plt.plot(X_train[y_train][i])
    plt.savefig(out_dir / f'norm_sig_data_{suffix:s}.png')

    if args.use_derivative:
        # we compute the derivative and also use it for classification
        dX_train = sample_derivative(X_train)
        dX_val = sample_derivative(X_val)
        X_train = np.dstack([X_train, dX_train])
        X_val = np.dstack([X_val, dX_val])
    else:
        # Univariate time series, i.e., only one feature for each time step
        # We need to transform the data so that the shape is (n_batch, n_time_steps, n_features)
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]

    print(X_train.shape)
    print(X_val.shape)

    # define new metric
    sm = SignificanceMetric(N_bkg_trigger=np.sum(~y_raw),
                            t_obs=t_tot_hrs * 3600.,
                            # dtype=tf.float
                            )
    # new way, assumes 20 days obs time, does not
    # take total number of samples, simply uses batch size
    metrics = list(models.metrics)
    metrics.append(sm)

    # set initial bias
    initial_bias, initial_loss = models.initial_output_bias(y_train)

    # define FCN model
    # Inspired by:
    # https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
    # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    # https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py

    if args.use_categorical:
        #model = build_model_test0b(X_train.shape[1:], metrics, output_bias=initial_bias)

        #filters = 8, 16
        #kernel_size = 7, 11
        #n_convs = 2
        #filters = 4, 8, 16
        # this worked ok for Extrinsics
        filters = 16
        kernel_size = 11
        n_convs = 2
        bn =True
        strides = 1

        # try for intrinsics:
        filters = 16
        kernel_size = 11
        n_convs = 2
        bn =True
        strides = 1


        # some conclusions: we need BN: faster convergence,
        # better performance
        # we need at least 2 convolutions, more convolutions
        # don't seem to help
        # kernel size up to 11 seems to perform better
        # filters > 16 don't seem to help much
        suffix += f"_nconvs{n_convs}_filters{filters}" \
                  f"_kernel_size{kernel_size}_strides{strides}_bn{bn:n}"
        model = build_model_test_modular(X_train.shape[1:],
                                         metrics,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         n_convs=n_convs,
                                         bn=bn,
                                         output_bias=initial_bias)
    else:
        model = build_model_test0(X_train.shape[1:], metrics, output_bias=initial_bias)
    #model = build_model_test1(X_train.shape[1:], metrics, output_bias=initial_bias)
    model.summary()
    initial_weights = out_dir / f'initial_weights'
    model.save_weights(initial_weights)
    plt.close("all")

    # do the fit
    if not (out_dir / "best_model{0:s}.hdf5".format(suffix)).exists() or args.overwrite:
        history = fit_fcn_model(model,
                                X_train,
                                to_categorical(y_train) if args.use_categorical else y_train,
                                X_val,
                                to_categorical(y_val) if args.use_categorical else y_val,
                                out_dir,
                                suffix=suffix,
                                epochs=epochs,
                                batch_size=batch_size)

    else:
        history = np.load(out_dir / "history{0:s}.npy".format(suffix), allow_pickle=True).item()
        model = keras.models.load_model(out_dir / "best_model{0:s}.hdf5".format(suffix),
                                        custom_objects=dict(SignificanceMetric=SignificanceMetric))

    # --- Plot performance --- #

    # plot the history 
    fig=plt.figure(figsize=(8,10))
    ax = plt.subplot(311)
    plot_metric(history, ax=ax, lw=2)
    ax.tick_params(direction='in', labelbottom=False)
    ax.set_xlabel("")
    ax.grid(which='both')
    plt.legend()

    ax = plt.subplot(312)
    plot_metric(history, ax=ax, metric='fp', lw=2)
    ax.tick_params(direction='in', labelbottom=False)
    ax.set_xlabel("")
    ax.grid(which='both')
    plt.legend()

    ax = plt.subplot(313)
    plot_metric(history, ax=ax, metric='significance', lw=2)
    ax.grid(which='both')
    plt.legend()

    plt.subplots_adjust(hspace=0.)
    plt.savefig(out_dir / f"metrics_vs_epochs_{suffix:s}.png")
    plt.close("all")

    # TODO optimize this with functions already existing
    results = {}
    threshold, significance_train, bkg_rate_train, eff_train = get_sig_vs_thr_keras(model,
                                                                                    X_train,
                                                                                    y_train,
                                                                                    t_obs_hours=t_tot_hrs,
                                                                                    #N_tot=y_train.size + \
                                                                                    #      y_val.size + sd.y_test.size,
                                                                                    N_tot=np.sum(~y_raw)
                                                                                    )
    results['thr_sig_bkg_eff_train'] = np.array([significance_train, bkg_rate_train, eff_train])

    _, significance_val, bkg_rate_val, eff_val= get_sig_vs_thr_keras(model,
                                                                     X_val,
                                                                     y_val,
                                                                     t_obs_hours=t_tot_hrs,
                                                                     #N_tot=y_train.size + y_val.size + sd.y_test.size,
                                                                     N_tot=np.sum(~y_raw)
                                                                     )

    results['thr_sig_bkg_eff_test'] = np.array([significance_val, bkg_rate_val, eff_val])
    results['thresholds'] = threshold

    # save output
    np.save(out_dir / "thr_sig_bkg_eff.npy", results)

    fig, ax_sig, ax_bkg, ax_eff = plot_performance_vs_threshold(thr=threshold,
                                                                bkg=bkg_rate_train,
                                                                eff=eff_train,
                                                                sig=significance_train,
                                                                label='Train',
                                                                t_tot_hours=t_tot_hrs,
                                                                rescale_t_obs_days=20. \
                                                                    if "extrinsic" in args.bkg_type else None,
                                                                )

    _ = plot_performance_vs_threshold(thr=threshold,
                                      bkg=bkg_rate_val,
                                      eff=eff_val,
                                      sig=significance_val,
                                      label='Val.',
                                      ax_bkg=ax_bkg,
                                      ax_eff=ax_eff,
                                      ax_sig=ax_sig,
                                      t_tot_hours=t_tot_hrs,
                                      rescale_t_obs_days=20. if "extrinsic" in args.bkg_type else None,
                                      fig=fig)
    id95 = np.argmin(np.abs(threshold - 0.95))
    logging.info(f"at thr={threshold[id95]:.3f}:")
    logging.info(f"(rescaled) significance val.={significance_val[id95]:.3f}")
    logging.info(f"bkg rate val.={bkg_rate_val[id95]:.3e}")
    logging.info(f"signal efficiency val.={eff_val[id95]:.3f}")

    if np.any(bkg_rate_train < 1e-7):
        vy = ax_bkg.get_ylim()
        ax_bkg.set_ylim(np.min([vy[0], 5e-8]), vy[1])
    if np.any(significance_train > 9.):
        vy = ax_sig.get_ylim()
        ax_sig.set_ylim(vy[0], np.max([vy[1], 11.]))
    plt.savefig(out_dir / f"sig_bkg_eff_vs_thr_{suffix:s}.png")
    plt.close("all")

    # plot class activation maps
    cam = calc_cam_binary_classification_utl(model,
                                             X_train,
                                             to_categorical(y_train) if args.use_categorical else y_train)
    pred = model.predict(X_train)
    pred_idx = np.argmax(pred, axis=1)


    n_samples = 15 # number of example time lines to plot
    cmap = plt.cm.coolwarm
    np.random.seed(args.random_state)
    # loop through classes
    for i_class, c in enumerate(cam):
        if args.use_categorical:
            if i_class:  # true positives
                m = y_train & pred_idx.astype(bool)
            else:  # true negatives
                m = ~y_train & ~pred_idx.astype(bool)

        else:  # only true light
            m = y_train

        fig = plot_cam_timeline(X_train, m, c, cmap=cmap, n_samples=n_samples)
        fig.savefig(out_dir / f"cam_class{i_class:n}_{suffix:s}.png", dpi=150)
        fig.savefig(out_dir / f"cam_class{i_class:n}_{suffix:s}.pdf", dpi=150)
        plt.close("all")

    # plot misidentified pulses
    # for training
    if args.use_categorical:
        fig, ax = plot_misided_timelines(model, X_train, y_train)
        if fig is not None:
            fig.savefig(out_dir / f"misidentified_train_{suffix:s}.png", dpi=150)
        plt.close("all")

        fig, ax = plot_misided_timelines(model, X_val, y_val)
        if fig is not None:
            fig.savefig(out_dir / f"misidentified_val_{suffix:s}.png", dpi=150)
        plt.close("all")
