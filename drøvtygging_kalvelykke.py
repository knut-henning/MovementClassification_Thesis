# -*- coding: utf-8 -*-
""" Drøvtygging maskinlæring
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'


import prepro_kalvelykke as pk
import visualize_kalvelykke as vk
import machinelearn_kalvelykke as mlk
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam



# Data prepro
act = pk.import_activity('atferd_kalvelykke\Egenklassifisert_atf_drøvtygging.csv')
act = pk.create_kalv(act)
act = pk.activity_set_datatypes(act)
act = pk.offset_time(act)
serials = pk.unique_serials(act)
start_stop = pk.activity_time_interval(act)
acc = pk.import_aks(serials, start_stop)
acc_act = pk.connect_data(act, acc)

# Visualize
vk.show_timestep_freq(pk.select_serial(acc_act, serials[0]))

vk.plot_acc(acc_act, serials)
vk.plot_acc(acc_act, serials, plot_all=False)

# Machinelearning prepro
# Data splitting
att = ['xcal', 'zcal']
mlk.replace_class(acc_act, {4: 1})
X_train, y_train, X_test, y_test = mlk.create_ratio_train_test(acc_act, serials, att)

# Standardize data
X_train, X_test = mlk.scale_norm(X_train, X_test, att, 'StandardScaler')

# Create timeseries
time_steps = 32
step = 31
X_train, y_train = mlk.create_timeseries(X_train, y_train, time_steps, step)
X_test, y_test = mlk.create_timeseries(X_test, y_test, time_steps, step)
print(X_train.shape, y_train.shape)

# One hot encode
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)
print(X_train.shape, y_train.shape)

# Machine learning
classes = ['No Chewing', 'Chewing']

# GRU
# Create model
model_gru = keras.Sequential()
model_gru.add(
    tf.keras.layers.Conv1D(
        filters=100, 
        kernel_size=3,
        activation='relu',
        input_shape=[X_train.shape[1], X_train.shape[2]]
    )
)
model_gru.add(tf.keras.layers.MaxPooling1D(pool_size=4, padding='valid'))
model_gru.add(tf.keras.layers.BatchNormalization())
model_gru.add(
    tf.keras.layers.Conv1D(
        filters=100, 
        kernel_size=3,
        activation='relu'
    )
)
model_gru.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid'))
model_gru.add(tf.keras.layers.BatchNormalization())
model_gru.add(
    tf.keras.layers.GRU(
        units=128,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        return_sequences=True,  # Next layer is also GRU
        dropout=0.0,
        recurrent_dropout=0.2
    )
)
model_gru.add(
    tf.keras.layers.GRU(
        units=128,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        return_sequences=False,  # Next layer is Dense
        dropout=0.2,
        recurrent_dropout=0.2
    )
)

model_gru.add(tf.keras.layers.Dropout(rate=0.1))

model_gru.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
model_gru.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])

# Fit model
history_gru = model_gru.fit(
    X_train, y_train,
    epochs=9,
    batch_size=64,
    validation_split=0.1,
    shuffle=False
)

model_gru.summary()

# Plot learning curve
vk.plot_learncurve(history_gru, 'GRU')

# Evaluate and plot confusion matrix
model_gru.evaluate(X_test, y_test)

y_test_n = np.argmax(y_test, axis=1) # Convert from onehot back to normal labels
y_pred_n = np.argmax(model_gru.predict(X_test), axis=-1)
vk.plot_matrix(y_test_n, y_pred_n, classes, 'GRU')


# LSTM
# Create model
model_lstm = tf.keras.Sequential()
model_lstm.add(
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            dropout=0.2,
            units=128, 
            input_shape=[X_train.shape[1], X_train.shape[2]],
            return_sequences=True
        )
    )
)
model_lstm.add(
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            dropout=0.2,
            units=64,
            return_sequences=False
        )
    )
)

model_lstm.add(tf.keras.layers.Flatten())
model_lstm.add(tf.keras.layers.Dense(units=5, activation='relu'))
model_lstm.add(tf.keras.layers.Dropout(rate=0.2))
model_lstm.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

# Fit model
history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    shuffle=False
)

model_lstm.summary()

# Plot learning curve
vk.plot_learncurve(history_lstm, 'LSTM')

# Evaluate ans plot confusion matrix
model_lstm.evaluate(X_test, y_test)

y_test_n = np.argmax(y_test, axis=1) # Convert from onehot back to normal labels
y_pred_n = np.argmax(model_lstm.predict(X_test), axis=-1)
vk.plot_matrix(y_test_n, y_pred_n, classes, 'LSTM')
