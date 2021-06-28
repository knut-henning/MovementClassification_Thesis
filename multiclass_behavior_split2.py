# -*- coding: utf-8 -*-
""" Multiclass behavior classification split2
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
# Data splitting
att = ['xcal', 'ycal', 'zcal', 'norm']
classes = ['Hviler', 'Bevegelse', 'Beiter', 'Dier']
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
                   "step": [15, 31],
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
    att, 'multi', 'split2', show_epochs=True,
    )


#%% Best params for GRU
# Reload model and data
multi_gru_losses = np.load('BestModels\\multi\\GRU\\split2\\val_losses.npy')

multi_gru_report = []

for i, model in enumerate(multi_gru_losses):
    multi_gru_model = keras.models.load_model('BestModels\\multi\\GRU\\split2\\model{}'.format(i))
    multi_gru_X_test = np.load('BestModels\\multi\\GRU\\split2\\model{}_Xtest.npy'.format(i))
    multi_gru_y_test = np.load('BestModels\\multi\\GRU\\split2\\model{}_ytest.npy'.format(i))

    # Model summary
    multi_gru_model.summary()
    
    # Evaluate and plot confusion matrix
    multi_gru_model.evaluate(multi_gru_X_test, multi_gru_y_test)

    y_test_n = np.argmax(multi_gru_y_test, axis=1) # Convert from onehot back to normal labels
    y_pred_n = np.argmax(multi_gru_model.predict(multi_gru_X_test), axis=-1)
    vp.plot_matrix(y_test_n, y_pred_n, classes, 'GRU', 'multi', show_title=False)
    multi_gru_report.append(vp.plot_pandas_classification_report(y_test_n, y_pred_n))


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
    att, 'multi', 'split2', show_epochs=True
    )


#%% Best params for LSTM
# Reload model and data
multi_lstm_losses = np.load('BestModels\\multi\\LSTM\\split2\\val_losses.npy')

multi_lstm_report = []

for i, model in enumerate(multi_lstm_losses):
    multi_lstm_model = keras.models.load_model('BestModels\\multi\\LSTM\\split2\\model{}'.format(i))
    multi_lstm_X_test = np.load('BestModels\\multi\\LSTM\\split2\\model{}_Xtest.npy'.format(i))
    multi_lstm_y_test = np.load('BestModels\\multi\\LSTM\\split2\\model{}_ytest.npy'.format(i))

    # Model summary
    multi_lstm_model.summary()
    
    # Evaluate and plot confusion matrix
    multi_lstm_model.evaluate(multi_lstm_X_test, multi_lstm_y_test)

    y_test_n = np.argmax(multi_lstm_y_test, axis=1) # Convert from onehot back to normal labels
    y_pred_n = np.argmax(multi_lstm_model.predict(multi_lstm_X_test), axis=-1)
    vp.plot_matrix(y_test_n, y_pred_n, classes, 'LSTM', 'multi', show_title=False)
    multi_lstm_report.append(vp.plot_pandas_classification_report(y_test_n, y_pred_n))
