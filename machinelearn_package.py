# -*- coding: utf-8 -*-
""" Machinelearning package
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


import pandas as pd
import numpy as np
import time
import tensorflow as tf
import itertools
import datetime
import json
from scipy import stats
from statistics import mean
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K


def import_data(path):
    """
    Import dataset.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    pd.DataFrame
        DataFrame with data.

    """
    return pd.read_csv(path, header=0)


def replace_class(data, val_dict):
    """
    Replaces classes in data with given keys in dict, example: {2: 1, 3: 2}.

    Parameters
    ----------
    data : pd.DataFrame
        Data to replace classes on.
    val_dict : dict
        Dictionary describing what classes to change and to what.

    Returns
    -------
    pd.DataFrame
        DataFrame with replaced classes.

    """
    return data['aktivitet'].replace(val_dict, inplace=True)


def create_ser_train_test(data, serials_train, serials_test, 
                          val_set, val_split, val_side, 
                          attributes):
    """
    Creates train and test data based on serials.

    Parameters
    ----------
    data : pd.DataFrame
        Data to split.
    serials_train : list
        Serials wanted for training data.
    serials_test : list
        Serials wanted for testing data.
    val_set : str
        Eider 'train' or 'test', what dataset to split validation set from
    val_split : float
        Precentage of training data to be validation data.
    val_side : str
        Eider 'left' or 'right' side of the training data to be split to validation data.
    attributes : list
        List of strings defining wanted attributes for machinelearning.

    Returns
    -------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.DataFrame
        Labels for training data.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.DataFrame
        Labels for validation data.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.DataFrame
        Labels for test data.

    """
    if type(serials_train) is list:
        df_train = pd.DataFrame(columns = data.columns)
        for train_serial in serials_train:
            df_train = df_train.append(data[data['serial'] == train_serial])
            
    elif isinstance(serials_train, (int, np.integer)):
        df_train = data[data['serial'] == serials_train]
    else:
        raise TypeError('serials in train must be a int or a list of integers')
        
    if type(serials_test) is list:
        df_test = pd.DataFrame(columns = data.columns)
        for test_serial in serials_test:
            df_test = df_test.append(data[data['serial'] == test_serial])
            
    elif isinstance(serials_train, (int, np.integer)):
        df_test = data[data['serial'] == serials_test]
    else:
        raise TypeError('serials in test must be a int or a list of integers')
    

    if val_set == 'train':
        n_obs_precentage = int(len(df_train) * val_split)
        
        X_test = df_test[attributes]
        y_test = df_test['aktivitet']
        
        if val_side == 'left':
            X_train = df_train[attributes].iloc[n_obs_precentage:]
            y_train = df_train['aktivitet'].iloc[n_obs_precentage:]
            
            X_val = df_train[attributes].iloc[:n_obs_precentage]
            y_val = df_train['aktivitet'].iloc[:n_obs_precentage]
            
        elif val_side == 'right':
            X_train = df_train[attributes].iloc[:-n_obs_precentage]
            y_train = df_train['aktivitet'].iloc[:-n_obs_precentage]
            
            X_val = df_train[attributes].iloc[-n_obs_precentage:]
            y_val = df_train['aktivitet'].iloc[-n_obs_precentage:]
            
        else:
            print('No side for validation split was assigned')

    elif val_set == 'test':
        n_obs_precentage = int(len(df_test) * val_split)
        
        X_train = df_train[attributes]
        y_train = df_train['aktivitet']
        
        if val_side == 'left':
            X_test = df_test[attributes].iloc[n_obs_precentage:]
            y_test = df_test['aktivitet'].iloc[n_obs_precentage:]
            
            X_val = df_test[attributes].iloc[:n_obs_precentage]
            y_val = df_test['aktivitet'].iloc[:n_obs_precentage]
            
        elif val_side == 'right':
            X_test = df_test[attributes].iloc[:-n_obs_precentage]
            y_test = df_test['aktivitet'].iloc[:-n_obs_precentage]
            
            X_val = df_test[attributes].iloc[-n_obs_precentage:]
            y_val = df_test['aktivitet'].iloc[-n_obs_precentage:]
            
        else:
            print('No side for validation split was assigned, "left" or "right"')
            
    else:
        print('No dataset was set for validation split, "train" or "test"')
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_ratio_train_val_test(data, serial, attributes, val_ratio=0.1, test_ratio=0.2, train_side='left'):
    """
    Creates train and test data based on one serial.

    Parameters
    ----------
    data : pd.DataFrame
        Data to split.
    serial : int
        Serial wanted for training data and test data.
    attributes : list
        List of strings defining wanted attributes for machinelearning.
    val_ratio : float, optional
        Ratio of data to use as validation data. The default is 0.1.
    test_ratio : float, optional
        Ratio of data to use as test data. The default is 0.2.
    train_side : str, optional
        Specify what side the training set is wanted.

    Returns
    -------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.DataFrame
        Labels for training data.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.DataFrame
        Labels for test data.

    """
    df = data[data['serial'] == serial[0]]
    
    df_len = len(df)
    test_int = int(df_len * test_ratio)
    val_int = int(df_len * val_ratio)
    train_int = int(df_len - test_int - val_int)
    
    if train_side == 'left':
        df_train = df.head(df_len - test_int - val_int)
        X_train = df_train[attributes]
        y_train = df_train['aktivitet']
        df_train_len = len(df_train)
        
        df_test = df.tail(test_int)
        X_test = df_test[attributes]
        y_test = df_test['aktivitet']
        df_test_len = len(df_test)
        
        df_val = df.iloc[df_train_len:(df_len-df_test_len)]
        X_val = df_val[attributes]
        y_val = df_val['aktivitet']
        
    elif train_side == 'right':
        df_test = df.head(df_len - train_int - val_int)
        X_test = df_test[attributes]
        y_test = df_test['aktivitet']
        df_test_len = len(df_test)
        
        df_train = df.tail(train_int)
        X_train = df_train[attributes]
        y_train = df_train['aktivitet']
        df_train_len = len(df_train)
        
        df_val = df.iloc[df_test_len:(df_len-df_train_len)]
        X_val = df_val[attributes]
        y_val = df_val['aktivitet']
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_timeseries(X, y, time_steps=1, step=1):
    """
    Creates sequenses of data, used for RNNs and other machine learning algoritms to group data.

    Parameters
    ----------
    X : pd.DataFrame
        Attributes with data you want to use for train/test.
    y : pd.DataFrame
        Lables for the data.
    time_steps : int, optional
        How many observations (rows) to include in each sequence. The default is 1.
    step : int, optional
        How many observations to skip before making next sequence. The default is 1.

    Returns
    -------
    np.array
        X with sequences and y.

    """
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
        
    return np.array(Xs), np.array(ys).reshape(-1, 1)


def scale_norm(X_train, X_val, X_test, attributes, scalertype):
    """
    Scales or normalize data based on wanted scaler/normalizer type.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training data.
    X_val : numpy.ndarray
        Training data.
    X_test : numpy.ndarray
        Test data.
    attributes : list
        List of attribute names for data.
    scalertype : str
        Scaler wanted for scaling.

    Raises
    ------
    TypeError
        Raises error if wrong scalertype is entered.

    Returns
    -------
    X_train : numpy.ndarray
        Training data scaled.
    X_val : numpy.ndarray
        Validation data scaled.
    X_test : numpy.ndarray
        Test data scaled.

    """
    if scalertype.lower() == 'standardscaler':
        scaler = StandardScaler()
    elif scalertype.lower() == 'robustscaler':
        scaler = RobustScaler()
    elif scalertype.lower() == 'minmaxscaler':
        scaler = MinMaxScaler()
    elif scalertype.lower() == 'maxabsscaler':
        scaler = MaxAbsScaler()
    elif scalertype.lower() == 'normalizer':
        scaler = Normalizer()
    else:
        raise TypeError('"scalerype" must be eider: "StandardScaler", "RobustScaler",'\
                        '"MinMaxScaler", "MaxAbsScaler" or "Normalizer"')
            
    scaler = scaler.fit(X_train)
    X_train.loc[:, attributes] = scaler.transform(X_train.to_numpy())
    X_val.loc[:, attributes] = scaler.transform(X_val.to_numpy())
    X_test.loc[:, attributes] = scaler.transform(X_test.to_numpy())
    
    return X_train, X_val, X_test


def create_hyp_configs(config_dict):
    """
    Generates all combinations of hyperparameteres defined in a input dict, also prints total 
    number of combinations.

    Parameters
    ----------
    config_dict : dict
        Dictionary with all hyperparameters as keys, and list of values for that hyperparameter
        as values.

    Returns
    -------
    configs : list
        List of all hyperparameter combinations.

    """
   
    keys, values = zip(*config_dict.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print('Total configs: {}'.format(len(configs)))
    
    return configs


def create_model_gru(X_train, y_train, X_val, y_val, X_test, y_test, gru_config, att, show_epochs, early_stop=False):
    """
    Creates a GRU model based on train/validation data and chosen hyperparams.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.DataFrame
        Labels for training data.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.DataFrame
        Labels for validation data.
    X_test : pd.DataFrame
        Testing data.
    y_test : pd.DataFrame
        Labels for testing data.
    gru_config : dict
        Chosen parameters for model
            Dict structure:
                "scaler": str
                "time_steps": iteger
                "step": iteger
                "conv1d_filters": iteger
                "dropout": float
                "gru_units": integer
                "learn_rate": float
                "epochs": integer
    att : list
        List of attributes to use.

    Returns
    -------
    model_gru : tensorflow.python.keras.engine.sequential.Sequential
        Returns fitted model with given hyperparameters.
    history_gru : tf.keras.callbacks.History
        Return history of scores for applied model.

    """
    
    # Standardize data
    X_train, X_val, X_test = scale_norm(X_train, X_val, X_test, att, gru_config.get("scaler"))
    
    # Create timeseries
    time_steps = gru_config.get("time_steps")
    step = gru_config.get("step")
    X_train, y_train = create_timeseries(X_train, y_train, time_steps, step)
    X_val, y_val = create_timeseries(X_val, y_val, time_steps, step)
    X_test, y_test = create_timeseries(X_test, y_test, time_steps, step)
    
    # One hot encode
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)
    
    # Create model
    model_gru = keras.Sequential()
    model_gru.add(
        tf.keras.layers.Conv1D(
            filters=gru_config.get("conv1d_filters"), 
            kernel_size=3,
            activation='relu',
            input_shape=[X_train.shape[1], X_train.shape[2]]
        )
    )
    model_gru.add(tf.keras.layers.MaxPooling1D(pool_size=4, padding='same'))
    model_gru.add(
        tf.keras.layers.Conv1D(
            filters=gru_config.get("conv1d_filters"), 
            kernel_size=3,
            activation='relu'
        )
    )
    model_gru.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'))
    
    model_gru.add(
        tf.keras.layers.GRU(
            units=gru_config.get("gru_units"),
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            return_sequences=True,  # Next layer is also GRU
        )
    )
    
    model_gru.add(
        tf.keras.layers.GRU(
            units=gru_config.get("gru_units"),
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            return_sequences=False,  # Next layer is Dense
        )
    )
    model_gru.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model_gru.compile(loss='categorical_crossentropy', optimizer=Adam(lr=gru_config.get("learn_rate")), metrics=['acc'])
    
    # Fit model
    if not show_epochs:
        v = 0
    elif show_epochs:
        v = 1
    
    if not early_stop:
        early_stop = None
    
    history_gru = model_gru.fit(X_train, y_train, 
                                epochs=gru_config.get("epochs"), 
                                batch_size=gru_config.get("batch_size"), 
                                validation_data=(X_val, y_val), 
                                shuffle=False, 
                                verbose=v,
                                callbacks=[early_stop]
                                )
    
    return model_gru, history_gru, X_test, y_test


def create_model_lstm(X_train, y_train, X_val, y_val, X_test, y_test, lstm_config, att, show_epochs, early_stop=False):
    """
    Creates a LSTM model based on train/validation data and chosen hyperparams.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.DataFrame
        Labels for training data.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.DataFrame
        Labels for validation data.
    X_test : pd.DataFrame
        Testing data.
    y_test : pd.DataFrame
        Labels for testing data.
    lstm_config : dict
        Chosen parameters for model
            Dict structure:
                "scaler": str
                "time_steps": iteger
                "step": iteger
                "lstm_units": iteger
                "dropout": float
                "dense_units": integer
                "learn_rate": float
                "epochs": integer
    att : list
        List of attributes to use.

    Returns
    -------
    model_lstm : tensorflow.python.keras.engine.sequential.Sequential
        Returns fitted model with given hyperparameters.
    history_lstm : tf.keras.callbacks.History
        Return history of scores for applied model.
    """
    
    # Standardize data
    X_train, X_val, X_test = scale_norm(X_train, X_val, X_test, att, lstm_config.get("scaler"))
    
    # Create timeseries
    time_steps = lstm_config.get("time_steps")
    step = lstm_config.get("step")
    X_train, y_train = create_timeseries(X_train, y_train, time_steps, step)
    X_val, y_val = create_timeseries(X_val, y_val, time_steps, step)
    X_test, y_test = create_timeseries(X_test, y_test, time_steps, step)
    
    # One hot encode
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)
    
    # Create model
    model_lstm = tf.keras.Sequential()
    model_lstm.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=lstm_config.get("lstm_units"), 
                input_shape=[X_train.shape[1], X_train.shape[2]],
                return_sequences=True
            )
        )
    )
    
    model_lstm.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=lstm_config.get("lstm_units"),
                return_sequences=True
            )
        )
    )
    
    model_lstm.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=lstm_config.get("lstm_units"),
                return_sequences=True
            )
        )
    )

    
    model_lstm.add(tf.keras.layers.Flatten())
    model_lstm.add(tf.keras.layers.Dense(units=lstm_config.get("dense_units"), activation='relu'))
    model_lstm.add(tf.keras.layers.Dropout(lstm_config.get("dropout")))
    model_lstm.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lstm_config.get("learn_rate")), metrics=['acc'])
    
    # Fit model
    if not show_epochs:
        v = 0
    elif show_epochs:
        v = 1
    
    if not early_stop:
        early_stop = None
        
        
    history_lstm = model_lstm.fit(X_train, y_train, 
                                  epochs=lstm_config.get("epochs"), 
                                  batch_size=lstm_config.get("batch_size"), 
                                  validation_data=(X_val, y_val), 
                                  shuffle=False, 
                                  verbose=v, 
                                  callbacks=[early_stop]
                                  )
    
    return model_lstm, history_lstm, X_test, y_test


def create_hyp_report(X_train, y_train, X_val, y_val, X_test, y_test, 
                      model, model_config_dict, att, dataset, split, show_epochs=True):
    """
    Generates a score report based on data and model configuration for hyperparameters. 
    (WARNING TUNING MAY TAKE A LONG TIME BASED ON NUMBER OF PARAM CONFIGURATIONS)

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.DataFrame
        Labels for training data.
    X_val : pd.DataFrame
        Validation data.
    y_val : pd.DataFrame
        Labels for validation data.
    X_test : pd.DataFrame
        Testing data.
    y_test : pd.DataFrame
        Labels for testing data.
    model : str
        Eider "GRU" or "LSTM" for what model to use
    model_config_dict : dict
        Chosen parameters for hyperparam tuning
            Dict structure:
                "scaler": list of str
                "time_steps": list of itegers
                "step": list of itegers
                "conv1d_filters/lstm_units": list of itegers
                "dropout": list of float between 0-1
                "gru_units/dense_units": list of integers
                "learn_rate": list of float
                "epochs": list of integers
    att : list
        List of attributes to use.
    dataset : str
        Name of dataset.
    split : str
        Name of split.
    show_epochs : boolean, optional
        Verbose on or off during training. The default is True.

    Returns
    -------
    None.
    """
    
    # Define callback function
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    
    # Generate configs
    configs = create_hyp_configs(model_config_dict)
    config_size = len(configs)
    
    dur_list = []
    if model.lower() == "gru":
        
        try:
            best_val_loss = list(np.load('BestModels\\{}\\GRU\\{}\\val_losses.npy'.format(dataset, split)))
        except FileNotFoundError:
            best_val_loss = [10, 11, 12, 13]
        
        for i, config in enumerate(configs):
            start = time.time()
            K.clear_session()
            
            gru_model, hist_gru, X_test_i, y_test_i = create_model_gru(
                X_train, y_train, 
                X_val, y_val, X_test, y_test, 
                config, att, 
                show_epochs, early_stop=early_stopping_monitor)
            
            model_max = max(best_val_loss)
            min_gru_val_loss = min(hist_gru.history.get("val_loss"))
            if min_gru_val_loss < model_max:
                best_value = min(hist_gru.history.get("val_loss"))
                best_val_loss[best_val_loss.index(model_max)] = best_value
                gru_model.save('BestModels\\{}\\GRU\\{}\\model{}'.format(dataset, split, best_val_loss.index(best_value)))
                np.save('BestModels\\{}\\GRU\\{}\\model{}_Xtest.npy'.format(dataset, split, best_val_loss.index(best_value)), 
                        X_test_i)
                np.save('BestModels\\{}\\GRU\\{}\\model{}_ytest.npy'.format(dataset, split, best_val_loss.index(best_value)), 
                        y_test_i)
                with open('BestModels\\{}\\GRU\\{}\\model{}_config.json'.format(dataset, split, best_val_loss.index(best_value)), 'w') as fp:
                    json.dump(hist_gru.history, fp)
                    json.dump(config, fp)
            
            stop = time.time()
            dur = stop-start  # Loop duration in seconds
            dur_list.append(dur) # Estimated time to completion in seconds
            est_dt = str(datetime.timedelta(seconds=mean(dur_list)*(config_size-i)))
            print('PROGRESS: {}/{} models | TIME: {:.2f}s | ESTIMATED TO COMPLETE: {}'.format(i+1, config_size, 
                                                                                              dur, est_dt))
            if i+1 == config_size:
                print("DONE! TIME SPENT: {}".format(str(datetime.timedelta(seconds=sum(dur_list))))) 
                print('The best validation loss: {}'.format(best_val_loss))
                
                np.save('BestModels\\{}\\GRU\\{}\\val_losses.npy'.format(dataset, split), np.array(best_val_loss))
            
    elif model.lower() == "lstm":
        
        try:
            best_val_loss = list(np.load('BestModels\\{}\\LSTM\\{}\\val_losses.npy'.format(dataset, split)))
        except FileNotFoundError:
            best_val_loss = [10, 11, 12, 13]
        
        for i, config in enumerate(configs):
            start = time.time()
            K.clear_session()

            lstm_model, hist_lstm, X_test_i, y_test_i = create_model_lstm(
                X_train, y_train, X_val, y_val, X_test, y_test,
                config, att, 
                show_epochs, early_stop=early_stopping_monitor)
            
            model_max = max(best_val_loss)
            min_lstm_val_loss = min(hist_lstm.history.get("val_loss"))
            if min_lstm_val_loss < model_max:
                best_value = hist_lstm.history.get("val_loss")[-2]
                best_val_loss[best_val_loss.index(model_max)] = best_value
                lstm_model.save('BestModels\\{}\\LSTM\\{}\\model{}'.format(dataset, split, best_val_loss.index(best_value)))
                np.save('BestModels\\{}\\LSTM\\{}\\model{}_Xtest.npy'.format(dataset, split, best_val_loss.index(best_value)), 
                        X_test_i)
                np.save('BestModels\\{}\\LSTM\\{}\\model{}_ytest.npy'.format(dataset, split, best_val_loss.index(best_value)), 
                        y_test_i)
                with open('BestModels\\{}\\LSTM\\{}\\model{}_config.json'.format(dataset, split, best_val_loss.index(best_value)), 'w') as fp:
                    json.dump(hist_lstm.history, fp)
                    json.dump(config, fp)

            
            stop = time.time()
            dur = stop-start  # Loop duration in seconds
            dur_list.append(dur) # Estimated time to completion in seconds
            est_dt = str(datetime.timedelta(seconds=mean(dur_list)*(config_size-i)))
            print('PROGRESS: {}/{} models | TIME: {:.2f}s | ESTIMATED TO COMPLETE: {}'.format(i+1, config_size, 
                                                                                              dur, est_dt))
            if i+1 == config_size:
                print("DONE! TIME SPENT: {}".format(str(datetime.timedelta(seconds=sum(dur_list)))))
                print('The best validation loss: {}'.format(best_val_loss))
                
                np.save('BestModels\\{}\\LSTM\\{}\\val_losses.npy'.format(dataset, split), np.array(best_val_loss))

            
    else:
        print('Please input model as str, "gru" or "lstm"')
    

    
if __name__ == '__main__':
    pass
