import numpy as np
import pandas as pd
import datetime
from utils.data_utils import resample_time, sliding_windows, split_dataset, generate_dataloader, normalise
from models import models

import torch
import torch.nn as nn
import torch.optim as optim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ANN')
parser.add_argument('--n_hist', type=int, default=7)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()
model_name = args.model
NUM_HIST, NUM_PRED = args.n_hist, args.n_pred
BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = args.batch_size, args.num_epochs, args.lr

# Reading data
# data = pd.read_csv('data/daily-total-female-births.csv')
raw_data = pd.read_csv('data/mpi_roof.csv', infer_datetime_format=True, parse_dates=['Date Time'])
data = resample_time(raw_data, 'Date Time', '4H')
time_seq = data['T (degC)'].values

# Preparing data, normalise, and train-val-test split
X, y = sliding_windows(time_seq, window=NUM_HIST, overlap=1, num_pred=NUM_PRED)
data_dict, MIN_VAL, MAX_VAL = normalise(split_dataset(X, y, test_size=0.2, seed=22))

# Generating dataloaders for modelling
train_loader = generate_dataloader(data_dict['train'][0], data_dict['train'][1], batch_size=BATCH_SIZE)
val_loader = generate_dataloader(data_dict['val'][0], data_dict['val'][1], batch_size=BATCH_SIZE)
test_loader = generate_dataloader(data_dict['test'][0], data_dict['test'][1], batch_size=BATCH_SIZE)

# Setting up model
if model_name == 'ANN':
    model = models.ANN(num_layers=2, num_nodes=[NUM_HIST, 64, 1])
elif model_name == 'LSTM':
    model = models.LSTM(num_layers=2, num_hidden=20, bidirectional=False)
elif model_name == 'GRU':
    model = models.GRU(num_layers=2, num_hidden=100)
elif model_name == 'RecursiveLSTM':
    model = models.RecursiveLSTM(num_pred=NUM_PRED, num_layers=1, num_hidden=10)

loss_fn = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=LEARNING_RATE)
print(f'Model architecture :\n{model}')

# train_iter = iter(train_loader)
# sample = train_iter.next()
# print(model(sample[0]))

# Train and validate the model
start_time = datetime.datetime.now()
for epoch in range(NUM_EPOCHS):
    train_loss = [0.0] * NUM_PRED
    val_loss = [0.0] * NUM_PRED

    model.train()
    for i, train_data in enumerate(train_loader):
        X_train, y_train = train_data
        optimiser.zero_grad()

        y_pred = model(X_train)
        with torch.no_grad():
            for p in range(NUM_PRED):
                loss = loss_fn(y_pred[:, p], y_train[:, p])
                train_loss[p] += loss

        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        for j, val_data in enumerate(val_loader):
            X_val, y_val = val_data
            for p in range(NUM_PRED):
                loss2 = loss_fn(model(X_val)[:, p], y_val[:, p])
                val_loss[p] += loss2

    print(f'Epoch {epoch + 1}, Training loss = {np.array(train_loss) / (i + 1)}, Validation loss = {np.array(val_loss) / (j + 1)}')
print(f'Training done, time taken is {(datetime.datetime.now() - start_time).seconds} seconds')

# Evaluating model on test set
model.eval()
with torch.no_grad():
    test_loss = [0.0] * NUM_PRED
    for k, test_data in enumerate(test_loader):
        X_test, y_test = test_data
        for p in range(NUM_PRED):
            loss3 = loss_fn(model(X_test)[:, p], y_test[:, p])
            test_loss[p] += loss3
print(f'Test loss = {np.array(test_loss) / (k + 1)}')

# # Saving the model
# torch.save(model.state_dict(), 'trained_models/' + model_name + '.pth')
# model = ANN(num_layers=3, num_nodes=[NUM_HIST, 64, 32, 1])
# model.load_state_dict(torch.load('trained_models/' + model_name + '.pth'))