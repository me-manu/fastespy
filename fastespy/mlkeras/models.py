from __future__ import absolute_import, division, print_function
from tensorflow import keras
import time
import numpy as np

metrics = (
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
)


def make_mlp_model(n_features,
                   metrics=metrics,
                   activation='relu',
                   output_bias=None,
                   learning_rate=3e-4,
                   n_layers=3,
                   n_nodes=100,
                   dropout=0.,
                   l2_regularizer=None):
    """
    Make a keras model for a multilayer perceptron
    for binary classification.

    Parameters
    ----------
    n_features: int
        number of features
    metrics: list
        list of metrics to be saved during training

    activation: string
        name of activation function

    output_bias: float
        initial bias for output layer

    learning_rate: float
        learning rate for Adam optimizer

    n_layers: int
        number of hidden layers

    n_nodes: int
        number of nodes in the hidden layers

    dropout: float
        Dropout rate

    l2_regularizer: float
        L2 Regularization parameter.

    Returns
    -------
    keras model instance

    """
    model = keras.Sequential(name="mlp_binary")

    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    if l2_regularizer is not None:
        l2_regularizer = keras.regularizers.l2(l2_regularizer)

    # adding the input layer
    model.add(keras.layers.Input(shape=(n_features)))

    # hidden layers
    for i in range(n_layers):
        model.add(keras.layers.Dense(n_nodes,
                                     activation=activation,
                                     name='dense{0:n}'.format(i + 1),
                                     kernel_regularizer=l2_regularizer,
                                     bias_regularizer=l2_regularizer
                                     )
                  )
        if dropout > 0.:
            model.add(keras.layers.Dropout(dropout))

    # output
    model.add(keras.layers.Dense(1, activation='sigmoid', name='output',
                                 bias_initializer=output_bias))  # output layer for binary classification

    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )

    return model


def initial_output_bias(y_train):
    """
    Compute the initial guess for the output bias
    for binary classification for imbalanced data

    Parameters
    ----------
    y_train: array-like
    Training class labels

    Returns
    -------
    tuple with array with initial guess for bias and resulting initial loss
    """
    initial_bias = np.array([np.log(y_train.sum() / np.invert(y_train.astype(bool)).astype(int).sum())])
    p0 = 1. / (1. + np.exp(-initial_bias))
    initial_loss = -p0 * np.log(p0) - (1. - p0) * np.log(1. - p0)
    return initial_bias, initial_loss


def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=200, batch_size=2048, **kwargs):
    """
    Train model

    Parameters
    ----------
    model: keras model
        the model to fit
    X_train: array-like
        training data
    y_train: array-like
        training labels
    X_val: None or array-like
        validation data
    y_val: None or array-like
        validation labels
    epochs: int
        training epocks
    batch_size: int
        batch size
    kwargs: dict
        addtional kwargs passed to fit funtion
    """
    kwargs.setdefault("verbose", 0)

    # early stopping if loss of validation set does not improve
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=10,
        mode='min',
        restore_best_weights=True)

    if X_val is None or y_val is None:
        valdation_data = None
    else:
        validation_data = (X_val, y_val)

    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        **kwargs
    )
    t1 = time.time()
    if kwargs['verbose']:
        print("training took {0:.2f}s".format(t1 - t0))

    return history
