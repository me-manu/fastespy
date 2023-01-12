#!/usr/bin/env python
# coding: utf-8

# # Test Tensorflow / Keras Models
# This notebook tests some machine learning on fitted TES data using Keras and Tensorflow

# In[1]:


import numpy as np
import glob
import os
import itertools
import matplotlib.pyplot as plt
import time
import copy
from fastespy.readpydata import convert_data_to_ML_format
from fastespy.plotting import plot_2d_hist, plot_scatter_w_hist
from fastespy.ml import MLHyperParTuning, significance
from fastespy.analysis import init_logging
from pathlib import PosixPath
import logging

import sys
sys.path.append("/Users/manuelmeyer/Python/fastespy/fastespy/scripts/")
from ml_intrinsic_bkg import load_data


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read data 

# In[3]:


path = "/Users/manuelmeyer/Downloads/IntrinsicsData_NewSetup/"
in_dir = PosixPath(path)
files = glob.glob(os.path.join(path, '*.npy'))

print(len(files))
#files


# In[4]:


init_logging("INFO", color=True)


# In[5]:


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

# read the data
result, data, t_tot_hrs = load_data(files, feature_names, light_cleaning_cuts=light_cleaning_cuts)

# convert data to ML format
X, y = convert_data_to_ML_format(result,
                                 feature_names,
                                 bkg_type=0,
                                 signal_type=1)


# In[6]:


X_log, y_log = MLHyperParTuning.transform_data_log(X.copy(), y.copy(), feature_names)


# In[7]:


k_folds = 5
ml = MLHyperParTuning(X_log, y_log,
                      valid_fraction=1. / k_folds,
                      stratify=True,
                      random_state=42,
                      n_splits=k_folds)

# now test and train data are contained 
# in ml.X_test, ml.X_train, ml.y_test, and ml.y_train


# To Do:
# - check whether PCA improves things
# - compare with sklearn results
# - compare with previous data set
# - more things to test: weighting, data augmentation, regularization with dropout?

# In[8]:


x = ml.kf.split(ml.X_train, ml.y_train)


# In[9]:


# generate train and validation set from stratified K fold

for train_index, val_index in ml._kf.split(ml.X_train, ml.y_train):
    
    # training data set
    X_train, y_train = ml.X_train[train_index], ml.y_train[train_index]
    
    # validation data set
    X_val, y_val = ml.X_train[val_index], ml.y_train[val_index]
    
# check class proportion
print(y_train.sum() / y_train.size)
print(y_val.sum() / y_val.size)
print(y_train.size)
print(y_val.size)
print(ml.y_test.size)


# ## Start with Keras and tensorflow

# In[10]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization


# In[11]:


# normalize the input data
normalizer = Normalization(axis=-1)

# adapt to training data
normalizer.adapt(X_train)

# normalize training data
X_train = normalizer(X_train)
X_val = normalizer(X_val)
X_test = normalizer(ml.X_test)


# In[12]:


print(np.var(X_train, axis=0))
print(np.var(X_val, axis=0))
print(np.var(X_test, axis=0))


# ### Building a first simple DNN

# In[13]:


# define different metrics
# see also https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
metrics = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

adam = keras.optimizers.Adam(learning_rate=3e-4)


# In[14]:


# initializing the sequential model
def make_model(metrics=metrics,
               output_bias=None,
               n_layers=3,
               n_nodes=100,
               l2_regularizer=None):
    
    model = keras.Sequential(name="dnn_test")
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    if l2_regularizer is not None:
        l2_regularizer = keras.regularizers.l2(l2_regularizer)
    
    # adding the input layer
    model.add(keras.layers.Input(shape=(X_train.shape[1])))
    
    # hidden layers
    for i in range(n_layers):
        model.add(keras.layers.Dense(n_nodes,
                                     activation='relu',
                                     name='dense{0:n}'.format(i + 1),
                                     kernel_regularizer=l2_regularizer,
                                     bias_regularizer=l2_regularizer
                                    )
                 )
    
    # output
    model.add(keras.layers.Dense(1, activation='sigmoid', name='output',
                                 bias_initializer=output_bias)) # output layer for binary classification
    
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=adam,
        metrics=metrics
    )
    
    return model


# In[15]:


n_nodes = 100
n_layers = 3
l2_regularizer = None


# In[16]:


# compute initial bias
# makes sense for imbalanced data, see the imbalanced data tutorial

initial_bias = np.array([np.log(y_train.sum() / np.invert(y_train.astype(bool)).astype(int).sum())])

print(initial_bias)
print(y_train.sum() / np.invert(y_train.astype(bool)).astype(int).sum())


# In[17]:


model = make_model(n_nodes=n_nodes,
                   n_layers=n_layers,
                   l2_regularizer=l2_regularizer,
                   output_bias=np.array([initial_bias])
                  )


# In[18]:


model.output_shape


# In[19]:


model.summary()


# In[20]:


EPOCHS = 100
BATCH_SIZE = 2048  # large enough so that you have enough signal samples in each batch

# early stopping if loss of validation set does not improve
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='min',
    restore_best_weights=True)


# In[21]:


# test the model
# with output bias, it should be roughly the class imbalance
print(y_train.sum()/ y_train.size)
model.predict(X_train[:10])


# In[22]:


results = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0, return_dict=True)


# In[23]:


# check the loss. if initial bias is correct, you should get something of the order of
# -ln(1 / n_classes) = ln(2)
print(np.log(2.), results['loss'])


# In[24]:


p0 = 1. / (1. + np.exp(-initial_bias))
print (-p0 * np.log(p0) - (1. - p0) * np.log(1. - p0))


# In[25]:


# how to check bias value
model.layers[-1].bias.value()


# In[26]:


import tempfile
import os


# Save the initial weights

# In[27]:


initial_weights = os.path.join(tempfile.mkdtemp(), f'initial_weights_{n_layers}_{n_nodes}')
model.save_weights(initial_weights)


# In[28]:


# check if new bias helps
model = make_model(n_nodes=n_nodes, n_layers=n_layers, l2_regularizer=l2_regularizer)
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(X_val, y_val), 
    verbose=0
)


# In[29]:


# now with bias
model = make_model(n_nodes=n_nodes, n_layers=n_layers, l2_regularizer=l2_regularizer)
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(X_val, y_val), 
    verbose=0
)


# In[30]:


# plot the results 
def plot_metric(history, metric="loss", **kwargs):
    label = kwargs.pop('label', '')
    plt.semilogy(history.epoch, history.history[metric], label='Train ' + label, **kwargs)
    
    kwargs.pop('ls', None)
    plt.semilogy(history.epoch, history.history[f'val_{metric}'], label='Val ' + label, ls='--', **kwargs)
    plt.xlabel('Epoch')
    plt.ylabel(metric)


# In[31]:


plot_metric(zero_bias_history, metric='loss', label='zero bias', color='C0')
plot_metric(careful_bias_history, metric='loss' , label='initial bias', color='C1')
plt.legend()


# The initial bias seems to help slightly.

# ### Train the full model

# In[32]:


# now with bias
model = make_model(n_nodes=n_nodes, n_layers=n_layers, l2_regularizer=l2_regularizer)
print(model.layers[-1].bias.value())
model.summary()
model.load_weights(initial_weights)
baseline_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val), 
    verbose=0
)


# In[33]:


plot_metric(baseline_history, color="C0")


# In[34]:


plot_metric(baseline_history, metric="fp", color="C0", label="false positives")
plot_metric(baseline_history, metric="fn", color="C1", label="false negatives")
plt.ylabel("Number of FP / FN")
plt.legend()


# In[35]:


# evaluate model on test set
test_scores = model.evaluate(X_test, ml.y_test, verbose=2, return_dict=True)
print("Test loss:", test_scores["loss"])
print("Test accuracy:", test_scores["accuracy"])
print(test_scores)


# In[36]:


y_pred_test = model.predict(X_test)
y_pred_val = model.predict(X_val)
y_pred_train = model.predict(X_train)


# In[37]:


threshold = 0.5 # threshold to classify event as "signal"
class_pred_test = (y_pred_test > threshold).flatten().astype(int)
class_pred_val = (y_pred_val > threshold).flatten().astype(int)
class_pred_train = (y_pred_train > threshold).flatten().astype(int)


# In[38]:


false_positive_test = (class_pred_test == 1) & (ml.y_test == 0)
false_negative_test = (class_pred_test == 0) & (ml.y_test == 1)

false_positive_val = (class_pred_val == 1) & (y_val == 0)
false_negative_val = (class_pred_val == 0) & (y_val == 1)

false_positive_train = (class_pred_train == 1) & (y_train == 0)
false_negative_train = (class_pred_train == 0) & (y_train == 1)


# In[39]:


print (false_positive_test.sum(), false_negative_test.sum())
print (false_positive_val.sum(), false_negative_val.sum())
print (false_positive_train.sum(), false_negative_train.sum())


# In[40]:


print("bkg rate test {0:.3e} Hz".format(
    false_positive_test.sum() / ml.y_test.size * (y_train.size + y_val.size + ml.y_test.size) / (t_tot_hrs * 3600.)))
print("bkg rate val {0:.3e} Hz".format(
    false_positive_val.sum() / ml.y_test.size * (y_train.size + y_val.size + ml.y_test.size) / (t_tot_hrs * 3600.)))
print("bkg rate train {0:.3e} Hz".format(
    false_positive_train.sum() / y_train.size * (y_train.size + y_val.size + ml.y_test.size) / (t_tot_hrs * 3600.)))


# ### Compute significance for threshold = 0.5

# In[41]:


from fastespy.ml import significance_scorer


# In[42]:


significance_scorer(ml.y_test, class_pred_test,
                    t_obs=t_tot_hrs * 3600.,
                    N_tot=y_train.size + y_val.size + ml.y_test.size)


# In[43]:


significance_scorer(y_val, class_pred_val, 
                    t_obs=t_tot_hrs * 3600.,
                    N_tot=y_train.size + y_val.size + ml.y_test.size)


# In[44]:


significance_scorer(y_train, class_pred_train,
                    t_obs=t_tot_hrs * 3600.,
                    N_tot=y_train.size + y_val.size + ml.y_test.size)


# In[45]:


# plot significance as function of threshold
def plot_sig_vs_thr(model, X, y_true, t_obs_hours, N_tot, step=0.002):
    y_pred = model.predict(X)
    
    threshold = np.arange(step, 1., step)
    significance = np.zeros_like(threshold)
    bkg_rate = np.zeros_like(threshold)
    eff = np.zeros_like(threshold)
    
    for i, thr in enumerate(threshold):
        class_pred = (y_pred > thr).flatten().astype(int)
    
        significance[i] = significance_scorer(y_true, class_pred,
                            t_obs=t_obs_hours * 3600.,
                            N_tot=N_tot)
        
        # bkg rate
        fp = (class_pred == 1) & (y_true == 0)
        fn = (class_pred == 0) & (y_true == 1)
        tp = (class_pred == 1) & (y_true == 1)
        bkg_rate[i] = fp.sum() / y_true.size * N_tot / (t_obs_hours * 3600.)
        eff[i] = tp.sum() / y_true.sum()
        
    ax = plt.subplot(311)
    ax.plot(threshold, significance)
    ax.set_ylabel("Significance ($\sigma$)")
    ax.tick_params(labelbottom=False, direction="in")
    ax.grid()
    
    ax = plt.subplot(312)
    ax.plot(threshold, bkg_rate)
    ax.set_yscale("log")
    ax.set_ylabel("Bkg rate (Hz)")
    ax.tick_params(labelbottom=False, direction="in")
    ax.grid()
    ax.set_ylim(5e-6, ax.get_ylim()[1])
        
    ax = plt.subplot(313)
    ax.plot(threshold, eff)
    #ax.set_yscale("log")
    ax.set_ylabel("Efficiency")
    ax.tick_params(direction="in")
    ax.grid()
    
    return ax


# In[46]:


fig=plt.figure(dpi=120, figsize=(6,2*3))
ax = plot_sig_vs_thr(model,
                     X_test,
                     ml.y_test,
                     t_obs_hours=t_tot_hrs,
                     N_tot=y_train.size + y_val.size + ml.y_test.size,
                     step=0.005
                     )
plt.subplots_adjust(hspace=0.)


# ### Plot misidentified pulses

# In[47]:


threshold = 0.99
class_pred_test = (y_pred_test > threshold).flatten().astype(int)
false_positive_test = (class_pred_test == 1) & (ml.y_test == 0)
false_negative_test = (class_pred_test == 0) & (ml.y_test == 1)


# #### False positives

# In[48]:


d = {}
d['y_test'] = ml.y_test
scorer = 'keras_loss'
d['y_pred_test'] = {scorer: class_pred_test}
d['idx_test'] = ml.idx_test
d['prob_test'] = {scorer: np.hstack([1. - y_pred_test, y_pred_test])}
d['classifier'] = "keras_mlp"

if false_positive_test.sum() < 50:
    ax = ml.plot_misidentified_time_lines(
        d, scorer,
        data['time'],
        data['data'],
        X=X,  # give original data, for right values in legend
        feature_names=feature_names,
        plot_false_positive=True, save_plot=False
    )


# #### False negatives

# In[49]:


d = {}
d['y_test'] = ml.y_test
scorer = 'keras_loss'
d['y_pred_test'] = {scorer: class_pred_test}
d['idx_test'] = ml.idx_test
d['prob_test'] = {scorer: np.hstack([1. - y_pred_test, y_pred_test])}
d['classifier'] = "keras_mlp"

if false_negative_test.sum() < 50:
    ax = ml.plot_misidentified_time_lines(
        d, scorer,
        data['time'],
        data['data'],
        X=X_log,  # give original data, for right values in legend
        feature_names=feature_names,
        plot_false_positive=False,
        save_plot=False
    )
else:
    print(f"False negatives too high for plotting: {false_negative_test.sum():n}")


# ## Try problem again with weighted class labels
# Weighting gives more "weight" to one class so that classifier pays more attention to it. 

# In[50]:


# weighting following
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.

weight_for_0 = 1 / (y.size - y.sum()) * (y.size / 2.0)
weight_for_1 = 1 / y.sum() * (y.size / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# In[51]:


weighted_model = make_model(n_nodes=n_nodes, n_layers=n_layers, l2_regularizer=l2_regularizer)
weighted_model.summary()
weighted_model.load_weights(initial_weights)


# In[ ]:


weighted_history = weighted_model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val), 
    verbose=0,
    class_weight=class_weight
)


# In[ ]:




