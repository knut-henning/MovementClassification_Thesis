# -*- coding: utf-8 -*-
""" Misc of plots for use in master thesis
"""

__author__ = 'Knut-Henning Kofoed'
__email__ = 'knut-henning@hotmail.com'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix

#%% Sigmoid
def sigmoid(z):
    """Compute logistic function (sigmoid)"""
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

z = np.arange(-8.,8.,0.1)
sig = sigmoid(z)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_xlabel('z')
ax.set_ylabel("$\phi(z)$")
ax.set_xlim(-9, 9)
ax.set_ylim(-0.1, 1.1)

plt.axhline(color='black', lw=0.5)
plt.axhline(y=0.5, color='grey', lw=0.5, ls='dotted')
plt.axhline(y=1, color='grey', lw=0.5, ls='dotted')
plt.axvline(color='black', lw=0.5)

plt.plot(z,sig)
plt.savefig('C:\\Users\\knut-\\OneDrive - Norwegian University of Life Sciences\\2021 Masteroppgave\\LaTeX\\Bilder\\Sigmoid.png')
plt.show()


#%% Hyperbolic Tangent
def tanh(z):
    """Compute logistic function (tanh)"""
    return np.tanh(z)

z = np.arange(-8.,8.,0.1)
ht = tanh(z)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_xlabel('z')
ax.set_ylabel("$\phi(z)$")
ax.set_xlim(-9, 9)
ax.set_ylim(-1.1, 1.1)

plt.axhline(color='black', lw=0.5)
plt.axhline(y=1, color='grey', lw=0.5, ls='dotted')
plt.axhline(y=-1, color='grey', lw=0.5, ls='dotted')
plt.axvline(color='black', lw=0.5)

plt.plot(z,ht)
plt.savefig('C:\\Users\\knut-\\OneDrive - Norwegian University of Life Sciences\\2021 Masteroppgave\\LaTeX\\Bilder\\Tanh.png')
plt.show()


#%% Rectified Linear Unit
def relu(z):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    z_list=[]
    for i in z:
        if i<0:
            z_list.append(0)
        else:
            z_list.append(i)

    return z_list

z = np.arange(-8.,8.,0.1)
y_relu = relu(z)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('z')
ax.set_ylabel("$\phi(z)$")

plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)

plt.plot(z,y_relu)
plt.savefig('C:\\Users\\knut-\\OneDrive - Norwegian University of Life Sciences\\2021 Masteroppgave\\LaTeX\\Bilder\\Relu.png')
plt.show()


#%% Confusion Matrix
plt.style.use('seaborn')
sns.set(font_scale=1.4)
t_labels = [random.randint(0,3) for _ in range(0,1000)]
p_labels = [random.randint(0,3) if j % 5 == 0 else i for j, i in enumerate(t_labels)]

classes = ['I ro','I bevegelse', 'Reiser seg', 'Legger seg']

cm = confusion_matrix(p_labels, t_labels)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', cmap="crest",
            xticklabels=classes, yticklabels=classes)
plt.ylabel('Sanne verdier', fontsize=18)
plt.xlabel('Predikerte verdier', fontsize=18)
plt.savefig('C:\\Users\\knut-\\OneDrive - Norwegian University of Life Sciences\\2021 Masteroppgave\\LaTeX\\Bilder\\ConfM.png')
plt.show(block=False)
