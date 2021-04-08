# -*- coding: utf-8 -*-


""" Preprocessing package
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
plt.style.use('seaborn')

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


def import_activity(data_path):
    """
    Imports activity data from given path.

    Parameters
    ----------
    data_path : str
        String with path and name of csv file to import.

    Returns
    -------
    pd.DataFrame
        Dataframe object of csv file. 

    """
    return pd.read_csv(data_path, header=0, delimiter=';', dtype=str)


def correction_activity(activity_data):
    """
    Used to correct the activity classified data provided by Nofence, contains some typos and
    unwanted spaces that needs correction before usage.

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity DataFrame to do correction on.

    Returns
    -------
    activity_data : pd.DataFrame
        Corrected DataFrame.

    """
    # Correct column names
    columns = ['Nofence ID', 'ID', 'Type dyr', 'Tid', 'Aktivitet', 'Dier kalv', 'Kommentar']
    
    # Sets the correct column names
    activity_data.columns = columns
    
    # Removes unwanted spaces to the right of the words.
    activity_data['Aktivitet'] = activity_data['Aktivitet'].str.rstrip()
    activity_data['Type dyr'] = activity_data['Type dyr'].str.rstrip()

    # Correct typos in column "Aktivitet"
    activity_data['Aktivitet'] = activity_data['Aktivitet'].replace({'Beter slutt': 'Beiter slutt'})
    
    # Removes rows that contain the word "Aktivitet" in the column "Aktivitet"
    activity_data = activity_data[~activity_data['Aktivitet'].str.contains('Aktivitet')]
    activity_data = activity_data.reset_index(drop=True)
    
    # Removes rows that contain the word "ny klave" in the column "Nofence ID"
    activity_data = activity_data[~activity_data['Nofence ID'].str.contains('ny klave')]
    activity_data = activity_data.reset_index(drop=True)

    return activity_data


def create_kalv(activity_data):
    """
    Creates a new column that specifies if the activity is from a calf or cow.
    "Kalv" (calf) True or False

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity dataframe to create "Kalv" column on.

    Returns
    -------
    pd.DataFrame
        Activity dataframe with added column "Kalv"

    """
    # Finds instances in "Type dyr" that contains "kalv", sets column value to True
    activity_data['Kalv'] = activity_data['Type dyr'].map(lambda x: 'kalv' in x)
    
    return activity_data


def activity_set_datatypes(activity_data):
    """
    Sets different datatypes for activity dataframe.

    Parameters
    ----------
    activity_data : pd.DataFrame
        Dataframe to set datatypes for

    Returns
    -------
    activity_data : pd.Dataframe
        Dataframe with set datatypes

    """
    # Removes ' from the datetime string (occurs in Nofence provided activity data)
    activity_data['Tid'] = activity_data['Tid'].str.rstrip("'")
    
    # Convert 'Tid' from 'str' to 'datetime'
    activity_data['Tid'] = pd.to_datetime(activity_data['Tid'])
    
    # Convert "Nofence ID" type from "str" to "int64"
    activity_data['Nofence ID'] = activity_data['Nofence ID'].astype('int64')
    
    return activity_data

    
def offset_time(activity_data, column='Tid', offset=-2):
    """
    Offset time of datetime column. (mainly used to convert datetime from CEST to UTC)

    Parameters
    ----------
    activity_data : pd.DataFrame
        Dataframe to offset datetime on
    column : str, optional
        Name of column to do the offset on. The default is 'Tid'.
    offset : int, optional
        Number of hours to offset. The default is -2.

    Returns
    -------
    activity_data : pd.DataFrame
        DataFrame with offset datetime values.

    """
    activity_data[column] = activity_data[column] + pd.DateOffset(hours=offset)
    
    return activity_data


def start_stop_corr(activity_data):
    """
    Mainly used for activity classification data provided by Nofence. For the later functions to work activity
    registration has to contain blocks with "VIDEO START" and "VIDEO SLUTT" to work. This function also prints how
    many rows has missing blocks with these strings.

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity registration data.

    Returns
    -------
    activity_data : pd.DataFrame
        Corrected for "VIDEO START" and "VIDEO SLUTT".

    """
    missing = 0  # Variable that stores number of rows that do not contain START/SLUTT in "Aktivitet" column
    row_iterator = activity_data.iterrows()
    _, last = next(row_iterator)  # First value of row_iterator
    
    for i, row in row_iterator:
        #  Saves the index where "VIDEO START" AND "VIDEO SLUTT" is expected in the "Aktivitet" column
        if (row['Nofence ID'] != last['Nofence ID']) & \
           (row['Aktivitet'] != 'VIDEO START') & \
           (last['Aktivitet'] != 'VIDEO SLUTT'):
               
            df = pd.concat([pd.DataFrame({"Nofence ID": last['Nofence ID'],
                                          "ID": last['ID'], 
                                          "Type dyr": last['Type dyr'],
                                          "Tid": last['Tid'],
                                          "Aktivitet": 'VIDEO SLUTT',
                                          "Kommentar": '',
                                          "Kalv": last['Kalv']},
                                         index = [i + missing]
                                        ),
                            pd.DataFrame({"Nofence ID": row['Nofence ID'],
                                          "ID": row['ID'],
                                          "Type dyr": row['Type dyr'],
                                          "Tid": row['Tid'],
                                          "Aktivitet": 'VIDEO START',
                                          "Kommentar": '',
                                          "Kalv": row['Kalv']},
                                         index = [i + missing + 1]
                                        )
                           ])
            activity_data = pd.concat([activity_data.iloc[:df.index[0]],
                                       df,
                                       activity_data.iloc[df.index[0]:]
                                       ]).reset_index(drop=True)
            missing += 2
    
            
        #  Saves the index where "VIDEO START" is expected in the "Aktivitet" column
        elif (row['Nofence ID'] != last['Nofence ID']) & \
             (row['Aktivitet'] != 'VIDEO START') & \
             (last['Aktivitet'] == 'VIDEO SLUTT'):
                 
            df = pd.DataFrame({"Nofence ID": row['Nofence ID'],
                               "ID": row['ID'],
                               "Type dyr": row['Type dyr'],
                               "Tid": row['Tid'],
                               "Aktivitet": 'VIDEO START',
                               "Kommentar": '',
                               "Kalv": row['Kalv']},
                              index = [i + missing]
                             )
            activity_data = pd.concat([activity_data.iloc[:df.index[0]],
                                       df,
                                       activity_data.iloc[df.index[0]:]
                                       ]).reset_index(drop=True)
            missing += 1
            
            
        #  Saves the index where "VIDEO SLUTT" is expected in the "Aktivitet" column
        elif (row['Nofence ID'] != last['Nofence ID']) & \
             (last['Aktivitet'] != 'VIDEO SLUTT') & \
             (row['Aktivitet'] == 'VIDEO START'):
                 
            df = pd.DataFrame({"Nofence ID": last['Nofence ID'],
                               "ID": last['ID'],
                               "Type dyr": last['Type dyr'],
                               "Tid": last['Tid'],
                               "Aktivitet": 'VIDEO SLUTT',
                               "Kommentar": '',
                               "Kalv": last['Kalv']},
                              index = [i + missing]
                             )
            activity_data = pd.concat([activity_data.iloc[:df.index[0]],
                                       df,
                                       activity_data.iloc[df.index[0]:]
                                       ]).reset_index(drop=True)
            missing += 1
            
        last = row
    
    
    #  Checks if the last row contains "VIDEO SLUTT" in the column "Aktivitet" 
    if row['Aktivitet'] != 'VIDEO SLUTT':
        df = pd.DataFrame({"Nofence ID": row['Nofence ID'],
                           "ID": row['ID'],
                           "Type dyr": row['Type dyr'],
                           "Tid": row['Tid'],
                           "Aktivitet": 'VIDEO SLUTT',
                           "Kommentar": '',
                           "Kalv": row['Kalv']},
                          index = [i + missing + 1]
                         )
        activity_data = pd.concat([activity_data.iloc[:df.index[0]],
                                   df,
                                   activity_data.iloc[df.index[0]:]
                                   ]).reset_index(drop=True)
        missing += 1
    
    print('Activity dataframe have {} missing rows with "VIDEO START/SLUTT"'.format(missing))
    
    return activity_data


def unique_serials(activity_data):
    """
    Creates a list of unique serials that the "Nofence ID" column in dataframe contains.

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity registration data.

    Returns
    -------
    serials : list
        List of serials.

    """
    serials = list(activity_data['Nofence ID'].unique())
    print('Serials from dataframe: {}'.format(serials))
    return serials


def activity_time_interval(activity_data):
    """
    Makes a dataframe with all "VIDEO START" "VIDEO SLUTT" intervals

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity registration data.

    Returns
    -------
    start_stop : pd.DataFrame
        Rows containing "VIDEO START/SLUTT"

    """
    start_stop = activity_data[(activity_data['Aktivitet'] == 'VIDEO START') | \
                               (activity_data['Aktivitet'] == 'VIDEO SLUTT')]
    return start_stop


def acc_time_corr(acc_data):
    """
    Gives the data better time resolution since it originally only updated every 32 observation.

    Parameters
    ----------
    acc_data : pd.DataFrane
        Accelerometer data.

    Returns
    -------
    acc_data : pd.DataFrame
        Accelerometer data with better time resolution.

    """
    times = acc_data[['date', 'header_date']]
    unique_time = times.drop_duplicates(subset=['header_date'])
    unique_time['time_delta'] = unique_time['header_date'] - unique_time['header_date'].shift()
    unique_time = unique_time.append({'time_delta': timedelta(seconds = 3)}, ignore_index=True)
    
    time_iterator = unique_time.iterrows()
    _, last = next(time_iterator)  # First value of time_iterator
    for i, time in time_iterator:
        dt = time['time_delta'].total_seconds()
        dt = dt / 32
        df_dt = pd.to_timedelta(acc_data['index'].iloc[(i-1)*32:32+((i-1)*32)] * dt, unit='s')
        acc_data['header_date'].iloc[(i-1)*32:32+((i-1)*32)] = acc_data['header_date'].iloc[(i-1)*32:32+((i-1)*32)] \
            + df_dt
        acc_data['header_date'] = acc_data.header_date.dt.ceil(freq='s')  
        
    return acc_data


def import_aks(serials, start_stop):
    """
    Sort relevant accelerometerdata based on activity registration data.

    Parameters
    ----------
    serials : list
        List of serial numbers you want accelerometerdata from.
    start_stop : pd.DataFrame
        Dataframe containing activity registration intervals.

    Returns
    -------
    start_slutt_acc : pd.DataFrame
        Accelerometer data from the timeintervals and serials expressed in serials and start_stop input.

    """
    # Define column names
    start_slutt_acc = pd.DataFrame(columns=['serial', 'date', 'header_date', 'index',
                                            'x', 'y', 'z',
                                            'xcal','ycal', 'zcal',
                                            'norm'])
    for serial in serials:
        # Import files
        df_acc = pd.read_csv('akselerometer_kalvelykke\kalvelykke-{0}.csv'.format(serial), header=1)
        # Convert 'date' from str to datetime
        df_acc['header_date'] = pd.to_datetime(df_acc['header_date'], format='%Y-%m-%dT%H:%M:%S')
        
        # Makes a simple dataframe for all "VIDEO START/SLUTT" rows with selected serial
        start_stop_ID = start_stop[(start_stop["Nofence ID"] == serial)]
        
        # Makes simple dataframe for start and stop datetimes and combines to own interval dataframe
        start_ID = start_stop_ID[(start_stop_ID["Aktivitet"] == 'VIDEO START')]
        start_ID = start_ID['Tid'].reset_index(drop=True)
        stop_ID = start_stop_ID[(start_stop_ID["Aktivitet"] == 'VIDEO SLUTT')]
        stop_ID = stop_ID['Tid'].reset_index(drop=True)
        
        intervals = pd.concat([start_ID, stop_ID], axis=1)
        intervals.columns = ['start', 'stop']
        
        #  Combines all intervals to one dataframe with relevant data
        for i in intervals.index:
            df_interval = df_acc[(df_acc['header_date'] > intervals['start'][i]) & \
                                 (df_acc['header_date'] <= intervals['stop'][i])]
            df_interval = acc_time_corr(df_interval)
            start_slutt_acc = start_slutt_acc.append(df_interval,
                                                     ignore_index=True)
        
    return start_slutt_acc


def remove_dier(activity_data):
    """
    Removes the rows i activity registrations that contain "Dier start" or "Dier slutt and that has "Kalv" == False.

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity registration data.

    Returns
    -------
    activity_data : pd.DataFrame
        Activity registration data without "Dier start" and "Dier slutt" where "Kalv" == False.

    """
    activity_data = activity_data[(~activity_data['Aktivitet'].str.contains('Dier start')) | \
                                  (~activity_data['Aktivitet'].str.contains('Dier slutt')) & \
                                  (activity_data['Kalv'] == True)
                                  ]
    activity_data = activity_data.reset_index(drop=True)
    
    return activity_data


def connect_data(activity_data, start_slutt_acc):
    """
    Connects activity registrations and accelerometer data so that the accelerometer observations has lables.

    Parameters
    ----------
    activity_data : pd.DataFrame
        Activity registration data.
    start_slutt_acc : pd.DataFrame
        Accelerometer data

    Returns
    -------
    acc_act : pd.DataFrame
        Accelerometer data with lables.

    """
    # Activities: Resting = 0, Movement = 1, Grazing = 2, Suckle = 3, Ruminate = 4
    start_slutt_acc['kalv'] = np.nan
    start_slutt_acc['aktivitet'] = np.nan
    
    # Iterates through list of activity registrations
    acc = 0 # Start activity
    row_iterator = activity_data.iterrows()
    _, last = next(row_iterator)  # First Value of row_iterator
    for i, row in row_iterator:
        # Makes a mask for relevant timeinterval from accelerometer data that is to be labeled
        mask = (start_slutt_acc['serial'] == last['Nofence ID']) & \
               (start_slutt_acc['header_date'] > last['Tid']) & \
               (start_slutt_acc['header_date'] <= row['Tid'])
         
        if last['Aktivitet'] == 'VIDEO START':
            acc = 0 # All cases where the activity registration start the cow/calf is resting
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
            
        elif last['Aktivitet'] == 'VIDEO SLUTT':
            pass
            
        elif last['Aktivitet'] == 'Legger seg':
            acc = 0
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
            
        elif last['Aktivitet'] == 'Reiser seg':
            acc = 0
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
            
        elif last['Aktivitet'] == 'Dier start':
            start_slutt_acc.loc[mask, 'aktivitet'] = 3
            
        elif last['Aktivitet'] == 'Dier slutt':
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
            
        elif last['Aktivitet'] == 'Beiter start':
            start_slutt_acc.loc[mask, 'aktivitet'] = 2
            
        elif last['Aktivitet'] == 'Beiter slutt':
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
            
        elif last['Aktivitet'] == 'Bevegelse start':
            start_slutt_acc.loc[mask, 'aktivitet'] = 1
            
        elif last['Aktivitet'] == 'Bevegelse slutt':
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
        
        elif last['Aktivitet'] == 'Tygge start':
            start_slutt_acc.loc[mask, 'aktivitet'] = 4
        
        elif last['Aktivitet'] == 'Tygge slutt':
            start_slutt_acc.loc[mask, 'aktivitet'] = acc
        
        start_slutt_acc.loc[mask, 'kalv'] = last['Kalv'] # Makes a column that informs if data is calf or not
        last = row
    
    # Data has floating point precision errors that need correcting
    acc_act = start_slutt_acc.round({'xcal': 3, 'ycal': 3, 'zcal': 3})
    
    # Removes rows containing nan and converts the column "aktivitet" from float to int
    acc_act = acc_act.dropna()
    acc_act['aktivitet'] = acc_act['aktivitet'].astype('int64')
    
    return acc_act


def select_serial(df_input, serial):
    """
    Selects data based on serial

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame to do selection on.
    serial : TYPE
        Serial to select.

    Returns
    -------
    df_output : TYPE
        Selected data based on serial.

    """
    df_output = df_input[df_input['serial'] == serial]
    return df_output


def save_dataframe(data, path, index=False):
    """
    Saves data to given path as csv.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be saved.
    path : str
        Location and file name of data to be saved.
    index : boolean, optional
        Specifies if index is wanted or not. The default is False.

    Returns
    -------
    None.

    """
    data.to_csv(path, index=index)


if __name__ == '__main__':
    pass

