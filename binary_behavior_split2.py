# -*- coding: utf-8 -*-
""" Binary behavior classification split2
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


import preprocessing_package as pp
import visualize_package as vp
import machinelearn_package as mlp
import numpy as np
import pandas as pd
from tensorflow import keras


# Data prepro
act = pp.import_activity('behavior\\behaviors.csv')

act_35396 = act[act['Nofence ID'] == 35396]
act_37368 = act[act['Nofence ID'] == 37368]
act_35396 = pp.offset_time(act_35396, column='Tid', hour=-2, finetune=True, second=19)
act_37368 = pp.offset_time(act_37368, column='Tid', hour=-2, finetune=False, second=0)
act = pd.concat([act_35396, act_37368])

serials = pp.unique_serials(act)
start_stop = pp.activity_time_interval(act)
acc = pp.import_aks(serials, start_stop)
acc_act = pp.connect_data(act, acc)

# Visualize
vp.show_timestep_freq(pp.select_serial(acc_act, serials[0]))
vp.show_serial_dist(acc_act)
vp.plot_acc(acc_act, serials)
vp.plot_acc(acc_act, serials, plot_all=False)


#%% Machinelearning preprocessing
# Change behaviors so that it is binary
mlp.replace_class(acc_act, {2: 1, 3: 1})

# Data splitting
att = ['xcal', 'ycal', 'zcal', 'norm']
classes = ['Hviler', 'Bevegelse']
X_train, y_train, X_val, y_val, X_test, y_test = mlp.create_ser_train_test(acc_act,
                                                                           serials[1],
                                                                           serials[0],
                                                                           'test',
                                                                           0.5,
                                                                           'right',
                                                                           att)

vp.plot_classbal_trainsplit(y_train, y_val, y_test, classes, show_title=True)

#%% GRU hyper param tuning multi-classification
gru_config_dict = {"scaler": ['standardscaler'],
                   "time_steps": [32, 64, 96],
                   "step": [15, 31, 62],
                   "conv1d_filters": [10, 20, 50, 100],
                   "gru_units": [16, 32, 64, 96],
                   "learn_rate": [0.001],
                   "epochs": [100],
                   "batch_size": [32, 64, 128]
                   }

mlp.create_hyp_report(
    X_train, y_train, 
    X_val, y_val, 
    X_test, y_test, 
    'gru', gru_config_dict,
    att, 'binary', 'split2', show_epochs=True
    )


#%% Best params for GRU
# Reload model and data
binary_gru_losses = np.load('BestModels\\binary\\GRU\\split2\\val_losses.npy')

binary_gru_report = []

for i, model in enumerate(binary_gru_losses):
    binary_gru_model = keras.models.load_model('BestModels\\binary\\GRU\\split2\\model{}'.format(i))
    binary_gru_X_test = np.load('BestModels\\binary\\GRU\\split2\\model{}_Xtest.npy'.format(i))
    binary_gru_y_test = np.load('BestModels\\binary\\GRU\\split2\\model{}_ytest.npy'.format(i))

    # Model summary
    binary_gru_model.summary()
    
    # Evaluate and plot confusion matrix
    binary_gru_model.evaluate(binary_gru_X_test, binary_gru_y_test)

    y_test_n = np.argmax(binary_gru_y_test, axis=1) # Convert from onehot back to normal labels
    y_pred_n = np.argmax(binary_gru_model.predict(binary_gru_X_test), axis=-1)
    vp.plot_matrix(y_test_n, y_pred_n, classes, 'GRU', 'binary', show_title=False)
    binary_gru_report.append(vp.plot_pandas_classification_report(y_test_n, y_pred_n))



#%% LSTM multi-classification
lstm_config_dict = {"scaler": ['standardscaler'],
                    "time_steps": [32, 64, 96],
                    "step": [15, 31, 62],
                    "lstm_units": [64, 96, 128],
                    "dropout": [0, 0.2],
                    "dense_units": [6, 10],
                    "learn_rate": [0.001],
                    "epochs": [100],
                    "batch_size": [32, 64, 128]
                    }

mlp.create_hyp_report(
    X_train, y_train, 
    X_val, y_val, 
    X_test, y_test, 
    'lstm', lstm_config_dict, 
    att, 'binary', 'split2', show_epochs=True
    )


#%% Best params for LSTM
# Reload model and data
binary_lstm_losses = np.load('BestModels\\binary\\LSTM\\split2\\val_losses.npy')

binary_lstm_report = []

for i, model in enumerate(binary_lstm_losses):
    binary_lstm_model = keras.models.load_model('BestModels\\binary\\LSTM\\split2\\model{}'.format(i))
    binary_lstm_X_test = np.load('BestModels\\binary\\LSTM\\split2\\model{}_Xtest.npy'.format(i))
    binary_lstm_y_test = np.load('BestModels\\binary\\LSTM\\split2\\model{}_ytest.npy'.format(i))

    # Model summary
    binary_lstm_model.summary()
    
    # Evaluate and plot confusion matrix
    binary_lstm_model.evaluate(binary_lstm_X_test, binary_lstm_y_test)

    y_test_n = np.argmax(binary_lstm_y_test, axis=1) # Convert from onehot back to normal labels
    y_pred_n = np.argmax(binary_lstm_model.predict(binary_lstm_X_test), axis=-1)
    vp.plot_matrix(y_test_n, y_pred_n, classes, 'LSTM', 'binary', show_title=False)
    binary_lstm_report.append(vp.plot_pandas_classification_report(y_test_n, y_pred_n))

