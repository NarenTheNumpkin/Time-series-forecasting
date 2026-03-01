import torch
import torch.nn as n
import numpy as np
import pandas as pd
from constants import *
from dataset import ForecastDataset

data = pd.read_csv(FILE)
y = data['T (degC)'].to_numpy().astype(np.float32)

data.drop(
    columns=['T (degC)', 'Date Time'], 
    inplace=True
) 

x = data.to_numpy().astype(np.float32)

samples = int(len(x) * 0.7)
X_train, X_test = x[:samples], x[samples:]
Y_train, Y_test = y[:samples], y[samples:]

x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
y_mean, y_std = Y_train.mean(axis=0), Y_train.std(axis=0)

X_train -= x_mean
X_train /= x_std
X_test -= x_mean
X_test /= x_std

Y_train -= y_mean
Y_train /= y_std
Y_test -= y_mean
Y_test /= y_std

train_dataset = ForecastDataset(X_train, Y_train)
test_dataset = ForecastDataset(X_test, Y_test)

