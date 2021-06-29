# -*- coding: utf-8 -*-
""" Unsupervised classification K-means
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'

import preprocessing_package as pp
import visualize_package as vp
import machinelearn_package as mlp
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans

#%% Unsupervised multiclass behaviors
# Data prepro
act = pp.import_activity('behavior\\behaviors.csv')

# Correct time to UTC
act_35396 = act[act['Nofence ID'] == 35396]
act_37368 = act[act['Nofence ID'] == 37368]
act_35396 = pp.offset_time(act_35396, column='Tid', hour=-2, finetune=True, second=19)
act_37368 = pp.offset_time(act_37368, column='Tid', hour=-2, finetune=False, second=0)
act = pd.concat([act_35396, act_37368])

# Connect activity observations with accelerometerdata
serials = pp.unique_serials(act)
start_stop = pp.activity_time_interval(act)
acc = pp.import_aks(serials, start_stop)
acc_act = pp.connect_data(act, acc)

# Visualize
vp.show_timestep_freq(pp.select_serial(acc_act, serials[0]))
vp.show_serial_dist(acc_act)
vp.plot_acc(acc_act, serials)
vp.plot_acc(acc_act, serials, plot_all=False)


# Machinelearning preprocessing

# Data splitting
att = ['xcal', 'ycal', 'zcal', 'norm']
classes = ['Hviler', 'Bevegelse', 'Beiter', 'Dier']
X_train, y_train, X_val, y_val, X_test, y_test = mlp.create_ser_train_test(acc_act,
                                                                           serials[0],
                                                                           serials[1],
                                                                           'test',
                                                                           0.001,
                                                                           'left',
                                                                           att)

vp.plot_classbal_trainsplit(y_train, y_val, y_test, classes, show_title=True)
X_train, X_val, X_test = mlp.scale_norm(X_train, X_val, X_test, att, "standardscaler")

time_steps = 64
step = 31

X_train_s, y_train_s = mlp.create_timeseries(X_train, y_train, time_steps, step)
X_val_s, y_val_s = mlp.create_timeseries(X_val, y_val, time_steps, step)
X_test_s, y_test_s = mlp.create_timeseries(X_test, y_test, time_steps, step)

# Machinelearning clustering
kmeans = TimeSeriesKMeans(n_clusters = 4, metric="dtw", verbose = True, random_state = 0, n_jobs=-1, max_iter=100).fit(X_train_s)

# Predict
y_pred_s = kmeans.predict(X_test_s)
vp.plot_matrix(y_test_s, y_pred_s, classes, 'Kmeans', 'multi')
vp.plot_pandas_classification_report(y_test_s, y_pred_s)

#%% Unsupervised binary behaviors
# Data prepro
act = pp.import_activity('behavior\\behaviors.csv')

# Correct time to UTC
act_35396 = act[act['Nofence ID'] == 35396]
act_37368 = act[act['Nofence ID'] == 37368]
act_35396 = pp.offset_time(act_35396, column='Tid', hour=-2, finetune=True, second=19)
act_37368 = pp.offset_time(act_37368, column='Tid', hour=-2, finetune=False, second=0)
act = pd.concat([act_35396, act_37368])

# Connect activity observations with accelerometerdata
serials = pp.unique_serials(act)
start_stop = pp.activity_time_interval(act)
acc = pp.import_aks(serials, start_stop)
acc_act = pp.connect_data(act, acc)

# Visualize
vp.show_timestep_freq(pp.select_serial(acc_act, serials[0]))
vp.show_serial_dist(acc_act)
vp.plot_acc(acc_act, serials)
vp.plot_acc(acc_act, serials, plot_all=False)


# Machinelearning preprocessing

# Change behaviors so that it is binary
mlp.replace_class(acc_act, {2: 1, 3: 1})

# Data splitting
att = ['xcal', 'ycal', 'zcal', 'norm']
classes = ['Hviler', 'Bevegelse']
X_train, y_train, X_val, y_val, X_test, y_test = mlp.create_ser_train_test(acc_act,
                                                                           serials[0],
                                                                           serials[1],
                                                                           'test',
                                                                           0.001,
                                                                           'left',
                                                                           att)

vp.plot_classbal_trainsplit(y_train, y_val, y_test, classes, show_title=True)
X_train, X_val, X_test = mlp.scale_norm(X_train, X_val, X_test, att, "standardscaler")

time_steps = 64
step = 31

X_train_s, y_train_s = mlp.create_timeseries(X_train, y_train, time_steps, step)
X_val_s, y_val_s = mlp.create_timeseries(X_val, y_val, time_steps, step)
X_test_s, y_test_s = mlp.create_timeseries(X_test, y_test, time_steps, step)

# Machinelearning clustering
kmeans = TimeSeriesKMeans(n_clusters = 2, metric="dtw", verbose = True, random_state = 0, n_jobs=-1, max_iter=100).fit(X_train_s)

# Predict
y_pred_s = kmeans.predict(X_test_s)
vp.plot_matrix(y_test_s, y_pred_s, classes, 'Kmeans', 'binary')
vp.plot_pandas_classification_report(y_test_s, y_pred_s)

#%% Unsupervised ruminant
# Data prepro
act = pp.import_activity('behavior\\ruminate.csv')

# Correct time to UTC
act = pp.offset_time(act, finetune=True, second=19)

# Connect activity observations with accelerometerdata
serials = pp.unique_serials(act)
start_stop = pp.activity_time_interval(act)
acc = pp.import_aks(serials, start_stop)
acc_act = pp.connect_data(act, acc)

# Visualize
vp.show_timestep_freq(pp.select_serial(acc_act, serials[0]))
vp.show_serial_dist(acc_act)
vp.plot_acc(acc_act, serials)
vp.plot_acc(acc_act, serials, plot_all=False)


# Machinelearning preprocessing

# Change behaviors so that it is binary
mlp.replace_class(acc_act, {4: 1})

# Data splitting
att = ['xcal', 'ycal', 'zcal', 'norm']
classes = ['Tygger ikke', 'Tygger']
X_train, y_train, X_val, y_val, X_test, y_test = mlp.create_ratio_train_val_test(acc_act, serials, att, val_ratio=0.05)

vp.plot_classbal_trainsplit(y_train, y_val, y_test, classes, show_title=True)
X_train, X_val, X_test = mlp.scale_norm(X_train, X_val, X_test, att, "standardscaler")

time_steps = 42
step = 20

X_train_s, y_train_s = mlp.create_timeseries(X_train, y_train, time_steps, step)
X_val_s, y_val_s = mlp.create_timeseries(X_val, y_val, time_steps, step)
X_test_s, y_test_s = mlp.create_timeseries(X_test, y_test, time_steps, step)

# Machinelearning clustering
kmeans = TimeSeriesKMeans(n_clusters = 2, metric="dtw", verbose = True, random_state = 0, n_jobs=-1, max_iter=100).fit(X_train_s)

# Predict
y_pred_s = kmeans.predict(X_test_s)
vp.plot_matrix(y_test_s, y_pred_s, classes, 'Kmeans', 'ruminant')
vp.plot_pandas_classification_report(y_test_s, y_pred_s)