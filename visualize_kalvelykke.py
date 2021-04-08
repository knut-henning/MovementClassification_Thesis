# -*- coding: utf-8 -*-

""" Visualizing package
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


from datetime import datetime
from sklearn.metrics import confusion_matrix
import numpy as np
import prepro_kalvelykke
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')


def show_df_structure(input_df):
    """
    Prints shape of data and its Layout.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame you want to print datastructure of

    Returns
    -------
    None.

    """
    
    print('Shape of data: {0} \n'.format(input_df.shape))
    print('Layout: \n {0} \n'.format(input_df))


def show_timestep_freq(input_df):
    """
    Prints TimeStep and UpdateFrequency of the input dataframe, 
    has to have a column named header_date with datetime of format
    %Y-%m-%dT%H:%M:%S
    
    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame you want to calculate frequency and timestep of

    Returns
    -------
    None.

    """
    
    try:
        n_obs = len(input_df)
        if str(input_df.dtypes[3]) == 'str':
            t1 = datetime.strptime(input_df['header_date'][0], '%Y-%m-%dT%H:%M:%S')
            t2 = datetime.strptime(input_df['header_date'][n_obs-1], '%Y-%m-%dT%H:%M:%S')
            
        else:
            t1 = input_df['header_date'].iloc[0]
            t2 = input_df['header_date'].iloc[-1]
            
        t_diff = t2 - t1
        dt = t_diff.total_seconds() / n_obs
        hz = n_obs / t_diff.total_seconds()
        print('TimeStep: {0}s'.format(dt))
        print('UpdateFrequency: {0}Hz'.format(hz))
    
    except TypeError as err1:
        print("Make sure input is a DataFrame. ERR:",
              err1)
        
    except KeyError as err2:
        print("Make sure DataFrame has col named header_date with format %Y-%m-%dT%H:%M:%S ERR:",
              err2)
    

def create_interval(acc_act):
    """
    Makes intervals to be able to visualize class distribution.

    Parameters
    ----------
    acc_act : pd.DataFrame
        Accelerometer data with lables.

    Returns
    -------
    i_interval : list
        Nested list with intervals and its class.

    """
    classes = acc_act['aktivitet'].unique()
    i_interval = []
    for c in classes:
        indexes = list(acc_act[acc_act['aktivitet'] == c].index)
        if len(indexes) != 0:
            i_iterator = iter(indexes)
            last = next(i_iterator)
            i_start = indexes[0]
            for i in i_iterator:
                if i != (last + 1):
                    i_interval.append([c, i_start, last])
                    i_start = i
            
                last = i
            i_interval.append([c, i_start, indexes[-1]])
    return i_interval


def plot_acc(acc_act, serials, plot_all=True, plot_attribute='norm'):
    """
    Plots all attributes for specified serials or plots chosen attribute for specified serials.

    Parameters
    ----------
    acc_act : pd.DataFrame
        Dataframe with data to plot.
    serials : list
        List with serials you want to plot from dataframe.
    plot_all : boolean, optional
        Plot all attributes or chosen one specified in variable plot_attribute. The default is True.
    plot_attribute : str, optional
        Attribute you want to plot. The default is 'norm'.

    Returns
    -------
    None.

    """
    colors = ['red', '#00ffe1', 'green', 'yellow', 'yellow']
    classes = acc_act['aktivitet'].unique()
    suptitles = ['Red: Resting', 'Light Blue: Movement', 'Green: Grazing', 'Yellow: Suckle', 'Yellow: Ruminate']
    suptitle = []
    for c in classes:
        suptitle.append(suptitles[c])
    
    if plot_all == True:
        # Visualize all serials with all attributes and with colored areas for different lables
        columns = ['x', 'y', 'z', 'xcal', 'ycal', 'zcal', 'norm']
        for serial in serials:
            data = prepro_kalvelykke.select_serial(acc_act, serial)
            i_interval = create_interval(data)
            
            fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
            plt.rcParams.update({'font.size': 10})
        
            linewidth = 0.4
            figsize = (12, 8)
            plt.suptitle('Serial: {}   Calf: {} \n {}'.format(
                         serial,
                         data['kalv'].unique(),
                         suptitle),
                fontweight ="bold")
        
            i_col = 0
            while i_col < len(columns):
                for row in range(3):
                    for col in range(3):
                        try:
                            data[columns[i_col]].plot(ax=axes[row,col], linewidth=linewidth, figsize=figsize)
                            axes[row, col].set_title(columns[i_col], fontweight="bold", size=8)
                            axes[row, col].tick_params(labelsize=6)
                        except IndexError:
                            break
                    
                        for interval in i_interval:
                            axes[row,col].axvspan(interval[1], interval[2],
                                                  color=colors[interval[0]],
                                                  alpha=0.4)
                        
                        i_col += 1
        
            fig.show()
        
    else:
        # Bigger picture of chosen attribute, default is 'norm'
        for serial in serials:
            data = prepro_kalvelykke.select_serial(acc_act, serial)
            i_interval = create_interval(data)
            plot_data = data[plot_attribute]
            
        
            plt.figure(figsize=(15, 6))
            plt.plot(plot_data, linewidth=0.4)
            for interval in i_interval:
                plt.axvspan(interval[1], interval[2],
                color=colors[interval[0]],
                alpha=0.4)
        
            plt.title('Serial: {}   Calf: {} \n {}'.format(
                      serial,
                      data['kalv'].unique(),
                      suptitle),
                fontweight ="bold",
                size=12)
            plt.tick_params(labelsize=10)
            plt.show()

    
def plot_obs_serials(data):
    """
    Plots oberservations per serial in data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.

    Returns
    -------
    None.

    """
    df_counts = data['serial'].value_counts()
    num_coll = len(df_counts) # Number of serials in set
    
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(range(num_coll), df_counts, width=0.5, align='center')
    ax.set(xticks=range(num_coll), xlim=[-1, num_coll])
    ax.set_xticklabels(df_counts.index)
    ax.set_xlabel('Serial ID')
    ax.set_ylabel('Number of observations')
    ax.set_title('Observations per serial')
    
    plt.show()


def plot_classbal_serials(data):
    """
    Plots class balance for each serial.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.

    Returns
    -------
    None.

    """
    serials = list(data['serial'].unique())

    for serial in serials:
        data = data['aktivitet'][data['serial'] == serial] 
        counts = np.bincount(data)
        n_classes = len(counts)
    
        fig, ax = plt.subplots(figsize=(14,5))
        ax.bar(range(n_classes), counts, width=0.5, align='center')
        ax.set(xticks=range(n_classes), xlim=[-1, n_classes])
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of observations')
        ax.set_title('Classbalance for accelerometer observations with serial: {}'.format(serial))
        plt.show()


def plot_learncurve(history, model):
    """
    Plots learning curves for loss and accuracy.

    Parameters
    ----------
    history : tensorflow.python.keras.callbacks.History
        History object created by model.fit.
    model : str
        Name of the model used (used for title).

    Returns
    -------
    None.

    """
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model: {}  Loss'.format(model))
    plt.legend();
    plt.show()
    
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.title('Model: {}  Acc'.format(model))
    plt.legend();
    plt.show()


def plot_matrix(y_test, y_pred, classes, model):
    """
    Plots Confusion Matrix, true labels vs predicted labels.

    Parameters
    ----------
    y_test : numpy.ndarray
        Series containing true labels.
    y_pred : numpy.ndarray
        Series containing predicted labels.
    classes : list
        List of the label names.
    model : str
        Name of the model used (used for title).

    Returns
    -------
    None.

    """
    cm = confusion_matrix(y_test, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap="crest",
                xticklabels=classes, yticklabels=classes)
    plt.title('Model: {}    Cells represent precentage predicted'.format(model))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    

if __name__ == '__main__':
    pass


