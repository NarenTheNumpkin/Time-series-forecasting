import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from constants import *
from dataset import ForecastDataset
from model import RecurrentNetwork
from train import Trainer

data = pd.read_csv(FILE)
y = data['T (degC)'].to_numpy().astype(np.float32)

data.drop(
    columns=['T (degC)', 'Date Time'], 
    inplace=True
) 

x = data.to_numpy().astype(np.float32)

samples = int(len(x) * SPLIT_RATIO)
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

def main():
    print("-------Loading Model-------\n")
    model = RecurrentNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("-------Loading Datasets-------\n")
    trainer_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Here we put shuffle=True because its batch shuffling and not row shuffling
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("mps" if torch.mps.is_available() else "cuda") # only MPS or GPU :D

    print("-------Loading Trainer-------\n")
    trainer = Trainer(model, optimizer, loss_fn, trainer_loader, val_loader, device, SAVES)
    trainer.train_epochs(EPOCHS)

if __name__ == "__main__":
    main()