from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import  tensorflow as tf
# Load a dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
print(dfeval.head())
y_train = dftrain.pop('survived')   # Removed from that dataset
y_eval = dfeval.pop('survived')
print(dftrain.head())
print(y_train)
print(dftrain.loc[0], y_train.loc[0])  # To find one specific row
print(dftrain["age"])
print(dftrain.describe())
print(dftrain.shape)