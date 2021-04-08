# -*- coding: utf-8 -*-
""" Machinelearning package
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer


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


def create_ser_train_test(data, serials_train, serials_test, attributes):
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
    attributes : list
        List of strings defining wanted attributes for machinelearning.

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
    if type(serials_train) is list:
        df_train = data[data['serial'].isin(serials_train)]
        df_test = data[data['serial'].isin(serials_test)]
    elif isinstance(serials_train, (int, np.integer)):
        df_train = data[data['serial'] == serials_train]
        df_test = data[data['serial'] == serials_test]
    else:
        raise TypeError('serials must be a int or a list of integers')
    
    X_train = df_train[attributes]
    y_train = df_train['aktivitet']
    
    X_test = df_test[attributes]
    y_test = df_test['aktivitet']
    
    return X_train, y_train, X_test, y_test


def create_ratio_train_test(data, serial, attributes, test_ratio=0.2):
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
    test_ratio : float, optional
        Ratio of data to use as test data. The default is 0.2.

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
    n_tail = int(df_len * test_ratio)
    
    df_train = df.head(df_len - n_tail)
    X_train = df_train[attributes]
    y_train = df_train['aktivitet']
    
    df_test = df.tail(n_tail)
    X_test = df_test[attributes]
    y_test = df_test['aktivitet']
    
    return X_train, y_train, X_test, y_test


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


def scale_norm(X_train, X_test, attributes, scalertype):
    """
    Scales or normalize data based on wanted scaler/normalizer type.

    Parameters
    ----------
    X_train : numpy.ndarray
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
    X_test.loc[:, attributes] = scaler.transform(X_test.to_numpy())
    
    return X_train, X_test


if __name__ == '__main__':
    pass
