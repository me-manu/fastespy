import numpy as np
import argparse
import logging
import time

from pathlib import PosixPath
from fastespy.utils import init_logging
from fastespy.io.readpydata import load_data_rikhav
import matplotlib.pyplot as plt

from fastespy.mlkeras import models
from fastespy.plotting import plot_metric
from tensorflow import keras
from fastespy.mlkeras.models import time_series_z_normalization
from fastespy.mlkeras.evaluation import (
     SplitData,
     SignificanceMetric
)


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


def build_ae_test0(input_shape, pool_size=4, n_filters=2, kernel_size=9, feature_vec_dim=7, learning_rate=1e-2):

    # build the encoder
    input_layer = keras.layers.Input(shape=(input_shape), name="input")
    conv1 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                padding='same', name="conv1", activation="relu")(input_layer)
    max_pool = keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size, padding='same')(conv1)
    flatten = keras.layers.Flatten()(max_pool)
    encoded = keras.layers.Dense(feature_vec_dim, activation='relu', name='feature_vector')(flatten)

    encoder = keras.models.Model(inputs=input_layer, outputs=encoded)

    # build the decoder
    n_nodes = int(input_shape[0] / pool_size * n_filters)
    dense = keras.layers.Dense(n_nodes, activation='relu', name='dense1')(encoded)
    reshape = keras.layers.Reshape((int(n_nodes / n_filters), n_filters))(dense)
    upsample = keras.layers.UpSampling1D(size=pool_size)(reshape)
    decoded = keras.layers.Conv1DTranspose(filters=1, padding="same", activation="relu",
                                           kernel_size=kernel_size)(upsample)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    autoencoder.compile(
        #loss=keras.losses.MeanSquaredError(),
        loss="mse",
        optimizer=adam
    )

    return autoencoder, encoder

def build_ae_test0b(input_shape, pool_size=4, n_filters=2, kernel_size=9, feature_vec_dim=7, learning_rate=1e-3):

    # build the encoder
    input_layer = keras.layers.Input(shape=(input_shape), name="input")
    encoded = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                padding='same', name="conv1", activation="relu")(input_layer)
    encoded = keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size, padding='same')(encoded)
    encoded = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                    padding='same', name="conv2", activation="relu")(encoded)
    encoded = keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size, padding='same')(encoded)

    encoded = keras.layers.Flatten()(encoded)
    encoded = keras.layers.Dense(feature_vec_dim, activation='relu', name='feature_vector')(encoded)

    encoder = keras.models.Model(inputs=input_layer, outputs=encoded)

    # build the decoder
    n_nodes = int(input_shape[0] / pool_size ** 2. * n_filters)

    decoded = keras.layers.Dense(n_nodes, activation='relu', name='dense1')(encoded)
    decoded = keras.layers.Reshape((int(n_nodes / n_filters), n_filters))(decoded)

    decoded = keras.layers.UpSampling1D(size=pool_size)(decoded)
    decoded = keras.layers.Conv1DTranspose(filters=n_filters, padding="same", activation="relu",
                                           kernel_size=kernel_size)(decoded)

    decoded = keras.layers.UpSampling1D(size=pool_size)(decoded)
    decoded = keras.layers.Conv1DTranspose(filters=1, padding="same", activation="relu",
                                           kernel_size=kernel_size)(decoded)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    autoencoder.compile(
        #loss=keras.losses.MeanSquaredError(),
        loss="mse",
        optimizer=adam
    )

    return autoencoder, encoder

def build_ae_test0c(input_shape, n_filters=16, kernel_size=11, feature_vec_dim=7, learning_rate=1e-3):

    # build the encoder
    input_layer = keras.layers.Input(shape=(input_shape), name="input")
    x = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                padding='same', name="conv1")(input_layer)
    #x = keras.layers.BatchNormalization(name=f"BN1")(x)
    x = keras.layers.Activation(activation='relu', name=f"relu1")(x)

    # 2nd conv + batch
    x = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                            padding='same', name="conv2")(x)
    #x = keras.layers.BatchNormalization(name=f"BN2")(x)
    x = keras.layers.Activation(activation='relu', name=f"relu2")(x)

    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    encoded = keras.layers.Dense(feature_vec_dim, activation='relu', name='feature_vector')(x)

    encoder = keras.models.Model(inputs=input_layer, outputs=encoded)

    # build the decoder
    decoded = keras.layers.Dense(n_filters, activation='relu', name='dense')(encoded)
    decoded = keras.layers.Reshape((1, n_filters))(decoded)
    decoded = keras.layers.UpSampling1D(size=input_shape[0])(decoded)

    decoded = keras.layers.Conv1DTranspose(filters=n_filters, padding="same", name="conv_t1",
                                           kernel_size=kernel_size)(decoded)
    #decoded = keras.layers.BatchNormalization(name=f"BN_t1")(decoded)
    decoded = keras.layers.Activation(activation='relu', name=f"relu_t1")(decoded)

    decoded = keras.layers.Conv1DTranspose(filters=1, padding="same", name="conv_t2",
                                           kernel_size=kernel_size)(decoded)
    #decoded = keras.layers.BatchNormalization(name=f"BN_t2")(decoded)
    decoded = keras.layers.Activation(activation='relu', name=f"relu_t2")(decoded)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    autoencoder.compile(
        #loss=keras.losses.MeanSquaredError(),
        loss="mse",
        optimizer=adam
    )

    return autoencoder, encoder

def build_ae_test1(input_shape, n_filters=32, kernel_size=7, strides=2, learning_rate=1e-3):
    """
    Model from https://keras.io/examples/timeseries/timeseries_anomaly_detection/
    """

    # build the encoder
    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    conv1 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                padding='same', name="conv1", activation="relu",
                                strides=strides)(input_layer)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    encoded = keras.layers.Conv1D(filters=int(n_filters / 2), kernel_size=kernel_size,
                                  padding='same', name="encoded", activation="relu",
                                  strides=strides)(conv1)
    encoder = keras.models.Model(inputs=input_layer, outputs=encoded)

    # build the decoder
    conv1_t = keras.layers.Conv1DTranspose(filters=int(n_filters / 2),
                                           padding="same", activation="relu",
                                           strides=strides,
                                           name="conv1_t",
                                           kernel_size=kernel_size)(encoded)
    conv1_t = keras.layers.Dropout(rate=0.2)(conv1_t)
    conv2_t = keras.layers.Conv1DTranspose(filters=n_filters,
                                           padding="same", activation="relu",
                                           strides=strides,
                                           name="conv2_t",
                                           kernel_size=kernel_size)(conv1_t)

    decoded = keras.layers.Conv1DTranspose(filters=input_shape[1],
                                           padding="same", activation="relu",
                                           name="decoded",
                                           kernel_size=kernel_size)(conv2_t)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    autoencoder.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=adam
    )

    return autoencoder, encoder

def build_ae_test2(input_shape,
                   pool_size=4,
                   n_filters=2,
                   kernel_size=9,
                   feature_vec_dim=7,
                   strides=2, learning_rate=1e-2):

    # build the encoder
    input_layer = keras.layers.Input(shape=(input_shape), name="input")

    conv1 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                 name="conv1", strides=strides)(input_layer)
    conv1 = keras.layers.BatchNormalization(name="BN1")(conv1)
    conv1 = keras.layers.Activation(activation='relu', name="relu1")(conv1)

    conv2 = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                name="conv2", strides=strides)(input_layer)
    conv2 = keras.layers.BatchNormalization(name="BN2")(conv1)
    conv2 = keras.layers.Activation(activation='relu', name="relu2")(conv1)

    max_pool = keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size, padding='same')(conv1)
    flatten = keras.layers.Flatten()(max_pool)
    encoded = keras.layers.Dense(feature_vec_dim, activation='relu', name='feature_vector')(flatten)

    encoder = keras.models.Model(inputs=input_layer, outputs=encoded)

    # build the decoder
    n_nodes = int(input_shape[0] / pool_size * n_filters)
    dense = keras.layers.Dense(n_nodes, activation='relu', name='dense1')(encoded)

    reshape = keras.layers.Reshape((int(n_nodes / n_filters), n_filters))(dense)
    reshape = keras.layers.BatchNormalization(name="BN2_t")(reshape)

    conv2_t = keras.layers.Conv1DTranspose(filters=n_filters,
                                           strides=strides,
                                           kernel_size=kernel_size)(reshape)
    conv2_t = keras.layers.BatchNormalization(name="BN1_1t")(conv2_t)
    conv2_t = keras.layers.Activation(activation='relu', name="relu2")(conv2_t)

    decoded = keras.layers.Conv1DTranspose(filters=1, padding="same", activation="relu", strides=strides,
                                           kernel_size=kernel_size)(conv2_t)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoded)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)

    autoencoder.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=adam
    )

    return autoencoder, encoder

def fit_ae_model(model, x_train, x_val, out_dir,
                  callbacks=None,
                  batch_size=2048,
                  epochs=50,
                  suffix='',
                  verbose=1):

    if callbacks is None:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                out_dir / "best_{0:s}.hdf5".format(suffix), save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]

    start_time = time.time()
    history = model.fit(x_train, x_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(x_val, x_val),
                        callbacks=callbacks)

    duration = time.time() - start_time

    model.save(out_dir / 'last_{0:s}.hdf5'.format(suffix))
    model = keras.models.load_model(out_dir / "best_{0:s}.hdf5".format(suffix))

    if verbose:
        print("training took {0:.2f}s".format(duration))

    # save training history
    np.save(out_dir / "history_{0:s}.npy".format(suffix), history.history)

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
    parser.add_argument('--random_state', help='Random state', type=int, default=42)

    args = parser.parse_args()
    init_logging("INFO", color=True)

    # get the files
    in_dir = PosixPath(args.indir)
    out_dir = PosixPath(args.outdir)
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
    #batch_size = 200
    batch_size = 100
    #epochs = 200
    epochs = 30   # for some testing
    factor_x = 1e6  # seconds to micro seconds
    factor_y = 1e3  # V to mV
    #downsample = 10  # use only every n-th data point / decrease sampling frequency by this amount
    #cut_start = int(data['time'].shape[1] * 0.05)
    #cut_end = int(data['time'].shape[1] * 0.4)

    # second try
    cut_start = int(data['time'].shape[1] * 0.1)
    cut_end = int(data['time'].shape[1] * 0.3)
    downsample = 4  # use only every n-th data point / decrease sampling frequency by this amount
    # 1000 samples is hard on my memory
    downsample = 1  # use only every n-th data point / decrease sampling frequency by this amount

    suffix = "ae_{5:s}sampling_freq{0:.1f}MHz_start{1:n}" \
             "_end{2:n}_batch_size{3:n}_epochs{4:n}".format(50. / downsample,
                                                            cut_start,
                                                            cut_end,
                                                            batch_size,
                                                            epochs,
                                                            args.suffix
                                                            )

    # define data
    X_raw = data['data'][:, cut_start:cut_end:downsample] * factor_y
    y_raw = result['type']

    # augment data
    #X, y = augment_imbalanced_binary_data(X_raw, y_raw, random_state=42, shuffle=True)
    X, y = X_raw, y_raw

    # split the data
    sd = SplitData(X, y, n_splits=5, stratify=True, random_state=42)

    # get one split, set aside some testing data
    i_split = 0
    X_train, X_val, y_train, y_val = sd.get_split_with_test_set(m=i_split, n=0)

    # z normalization for time series
    X_train = time_series_z_normalization(X_train, axis=1, invert=True, subtract_min=True)
    X_val = time_series_z_normalization(X_val, axis=1, invert=True, subtract_min=True)

    # plot some transformed data samples
    plt.figure(1)
    for i in range(10):
        plt.plot(X_train[~y_train][i])
    plt.savefig(out_dir / f'norm_bkg_data_{suffix:s}.png')

    # plot some transformed data samples
    plt.figure(2)
    for i in range(10):
        plt.plot(X_train[y_train][i])
    plt.savefig(out_dir / f'norm_sig_data_{suffix:s}.png')
    plt.close("all")

    # Univariate time series, i.e., only one feature for each time step
    # We need to transform the data so that the shape is (n_batch, n_time_steps, n_features)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # define new metric
    sm = SignificanceMetric(N_bkg_trigger=np.sum(~y_raw),
                            t_obs=t_tot_hrs * 3600.,
                            # dtype=tf.float
                            )
    metrics = list(models.metrics)
    metrics.append(sm)

    # define AE

    # initial setup inspired from paper:
    pool_size = 4
    kernel_size = 9
    n_filters = 2
    feature_vec_dim = 7

    # playing around
    pool_size = 4
    kernel_size = 11
    n_filters = 4
    feature_vec_dim = 10

    #autoencoder, encoder = build_ae_test0(X_train.shape[1:],
    #                                      pool_size=pool_size,
    #                                      kernel_size=kernel_size,
    #                                      n_filters=n_filters,
    #                                      feature_vec_dim=feature_vec_dim)

    n_filters = 2
    pool_size = 4
    feature_vec_dim = 10
    autoencoder, encoder = build_ae_test0b(X_train.shape[1:],
                                      pool_size=pool_size,
                                      kernel_size=kernel_size,
                                      n_filters=n_filters,
                                      feature_vec_dim=feature_vec_dim)
# This does not work at all
    #feature_vec_dim = 10
    #autoencoder, encoder = build_ae_test0c(X_train.shape[1:],
    #                                      kernel_size=11,
    #                                     n_filters=16,
    #                                    feature_vec_dim=feature_vec_dim)
    #
    # more filters help but lot more parameters. Peaks not too well sampled though
    #kernel_size = 7
    #n_filters = 32
    #strides = 2
    #autoencoder, encoder = build_ae_test1(X_train.shape[1:],
                                          #kernel_size=kernel_size,
                                          #n_filters=n_filters,
                                          #strides=strides
                                          #)

    #autoencoder, encoder = build_ae_test2(X_train.shape[1:],
                                          #kernel_size=kernel_size,
                                          #n_filters=n_filters,
                                          #strides=2,
                                          #feature_vec_dim=feature_vec_dim)
    autoencoder.summary()
    initial_weights = out_dir / f'initial_weights'
    autoencoder.save_weights(initial_weights)

    # do the fit
    if not (out_dir / "best_{0:s}.hdf5".format(suffix)).exists() or args.overwrite:
        history = fit_ae_model(autoencoder,
                               X_train,
                               X_val,
                               out_dir,
                               suffix=suffix,
                               epochs=epochs,
                               batch_size=batch_size)

    else:
        history = np.load(out_dir / "history_{0:s}.npy".format(suffix), allow_pickle=True).item()
        autoencoder = keras.models.load_model(out_dir / "best_{0:s}.hdf5".format(suffix),
                                        custom_objects=dict(SignificanceMetric=SignificanceMetric))

    # --- Plot performance --- #

    # plot the history 
    ax = plt.subplot(111)
    plot_metric(history, ax=ax, lw=2)
    ax.tick_params(direction='in', labelbottom=False)
    ax.set_xlabel("")
    ax.grid(which='both')
    plt.legend()
    plt.savefig(out_dir / f"loss_vs_epochs_{suffix:s}.png")
    plt.close("all")


    # --- Plot a reconstructed time line
    X_pred = autoencoder.predict(X_train)
    np.random.seed(args.random_state)
    size = 10

    idx = np.random.choice(range(X_train.shape[0]), size=size, replace=False)

    fig = plt.figure(figsize=(6, 10))
    for i_idx, i in enumerate(idx):
        y_shift = i_idx * -3
        x_shift = 0. #i_idx * -5
        plt.axhline(y_shift, ls=':', color="0.8")
        plt.plot(range(X_pred.shape[1]), X_pred[i, :, 0] + y_shift)
        plt.plot(range(X_pred.shape[1]), X_train[i, :, 0] + y_shift, ls='--')
    plt.savefig(out_dir / f"example_time_line_{suffix:s}.png")
    plt.close("all")

    # plot the histogram of mean squared error
    X_train_pred = autoencoder.predict(X_train)
    train_mse_loss = np.mean((X_train_pred - X_train) ** 2., axis=1)
    _ = plt.hist(train_mse_loss[y_train], bins=100, density=True, label="Signal", alpha=0.7)
    _ = plt.hist(train_mse_loss[~y_train], bins=100, density=True, label="Bkg", alpha=0.7)
    plt.legend()
    plt.savefig(out_dir / f"mse_hist_{suffix:s}.png")
    plt.close("all")

    # plot some time lines with min and max loss
    fig = plt.figure(figsize=(6,10))
    idx = np.argmin(train_mse_loss[~y_train, 0])
    plt.plot(X_train[~y_train][idx, :, 0], label="Min MSE Bkg data")
    plt.plot(X_train_pred[~y_train][idx, :, 0], label="Min MSE Bkg pred")

    y_shift = -3
    idx = np.argmax(train_mse_loss[~y_train, 0])
    plt.plot(X_train[~y_train][idx, :, 0] + y_shift, label="Max MSE Bkg data")
    plt.plot(X_train_pred[~y_train][idx, :, 0] + y_shift, label="Max MSE Bkg pred")

    y_shift = -6
    idx = np.argmin(train_mse_loss[y_train, 0])
    plt.plot(X_train[y_train][idx, :, 0] + y_shift, label="Min MSE Sig data")
    plt.plot(X_train_pred[y_train][idx, :, 0] + y_shift, label="Min MSE Sig pred")

    y_shift = -9
    idx = np.argmax(train_mse_loss[y_train, 0])
    plt.plot(X_train[y_train][idx, :, 0] + y_shift, label="Max MSE Sig data")
    plt.plot(X_train_pred[y_train][idx, :, 0] + y_shift, label="Max MSE Sig pred")
    plt.legend()
    plt.savefig(out_dir / f"mse_pulses_{suffix:s}.png")
    plt.close("all")

