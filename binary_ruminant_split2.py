# -*- coding: utf-8 -*-
""" Binary ruminant classification split2
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


import preprocessing_package as pp
import visualize_package as vp
import machinelearn_package as mlp
import numpy as np
from tensorflow import keras


# Data prepro
act = pp.import_activity('behavior\\ruminate.csv')
act = pp.offset_time(act, finetune=True, second=19)
serials = pp.unique_serials(act)
start_stop = pp.activity_time_interval(act)
acc = pp.import_aks(serials, start_stop)
acc_act = pp.connect_data(act, acc)

# Visualize
vp.show_timestep_freq(pp.select_serial(acc_act, serials[0]))
vp.show_serial_dist(acc_act)
vp.plot_acc(acc_act, serials)
vp.plot_acc(acc_act, serials, plot_all=False, plot_attribute='xcal')
vp.plot_acc(acc_act, serials, plot_all=False, plot_attribute='ycal')
vp.plot_acc(acc_act, serials, plot_all=False, plot_attribute='zcal')
vp.plot_acc(acc_act, serials, plot_all=False, plot_attribute='norm')


#%% Machinelearning preprocessing
# Change behaviors so that it is binary
mlp.replace_class(acc_act, {4: 1})

# Data splitting
att = ['xcal', 'ycal', 'zcal', 'norm']
classes = ['Tygger ikke', 'Tygger']
X_train, y_train, X_val, y_val, X_test, y_test = mlp.create_ratio_train_val_test(acc_act, serials, 
                                                                                 att, train_side='right')

vp.plot_classbal_trainsplit(y_train, y_val, y_test, classes, show_title=True)

#%% GRU hyper param tuning multi-classification
gru_config_dict = {"scaler": ['standardscaler'],
                   "time_steps": [11, 32, 64],
                   "step": [5, 10, 16, 31],
                   "conv1d_filters": [10, 50, 100],
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
    att, 'ruminant', 'split2', show_epochs=True
    )


#%% Best params for GRU
# Reload model and data
ruminant_gru_losses = np.load('BestModels\\ruminant\\GRU\\split2\\val_losses.npy')

ruminant_gru_report = []

for i, model in enumerate(ruminant_gru_losses):
    ruminant_gru_model = keras.models.load_model('BestModels\\ruminant\\GRU\\split2\\model{}'.format(i))
    ruminant_gru_X_test = np.load('BestModels\\ruminant\\GRU\\split2\\model{}_Xtest.npy'.format(i))
    ruminant_gru_y_test = np.load('BestModels\\ruminant\\GRU\\split2\\model{}_ytest.npy'.format(i))

    # Model summary
    ruminant_gru_model.summary()
    
    # Evaluate and plot confusion matrix
    ruminant_gru_model.evaluate(ruminant_gru_X_test, ruminant_gru_y_test)

    y_test_n = np.argmax(ruminant_gru_y_test, axis=1) # Convert from onehot back to normal labels
    y_pred_n = np.argmax(ruminant_gru_model.predict(ruminant_gru_X_test), axis=-1)
    vp.plot_matrix(y_test_n, y_pred_n, classes, 'GRU', 'ruminant', show_title=False)
    ruminant_gru_report.append(vp.plot_pandas_classification_report(y_test_n, y_pred_n))



#%% LSTM multi-classification
lstm_config_dict = {"scaler": ['standardscaler'],
                    "time_steps": [11, 32, 64],
                    "step": [5, 10, 16, 31],
                    "lstm_units": [16, 32, 64],
                    "dropout": [0, 0.2],
                    "dense_units": [3, 6, 10],
                    "learn_rate": [0.001],
                    "epochs": [100],
                    "batch_size": [32, 64, 128]
                    }

mlp.create_hyp_report(
    X_train, y_train, 
    X_val, y_val, 
    X_test, y_test, 
    'lstm', lstm_config_dict, 
    att, 'ruminant', 'split2', show_epochs=True
    )


#%% Best params for LSTM
# Reload model and data
ruminant_lstm_losses = np.load('BestModels\\ruminant\\LSTM\\split2\\val_losses.npy')

ruminant_lstm_report = []

for i, model in enumerate(ruminant_lstm_losses):
    ruminant_lstm_model = keras.models.load_model('BestModels\\ruminant\\LSTM\\split2\\model{}'.format(i))
    ruminant_lstm_X_test = np.load('BestModels\\ruminant\\LSTM\\split2\\model{}_Xtest.npy'.format(i))
    ruminant_lstm_y_test = np.load('BestModels\\ruminant\\LSTM\\split2\\model{}_ytest.npy'.format(i))

    # Model summary
    ruminant_lstm_model.summary()
    
    # Evaluate and plot confusion matrix
    ruminant_lstm_model.evaluate(ruminant_lstm_X_test, ruminant_lstm_y_test)

    y_test_n = np.argmax(ruminant_lstm_y_test, axis=1) # Convert from onehot back to normal labels
    y_pred_n = np.argmax(ruminant_lstm_model.predict(ruminant_lstm_X_test), axis=-1)
    vp.plot_matrix(y_test_n, y_pred_n, classes, 'LSTM', 'ruminant', show_title=False)
    ruminant_lstm_report.append(vp.plot_pandas_classification_report(y_test_n, y_pred_n))
