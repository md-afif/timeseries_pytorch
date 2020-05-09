import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils

def sliding_windows(x, window=7, overlap=1, num_pred=1):
    """
    Given a univariate time series, form time windows based on specified inputs
    e.g. window=3, num_pred=1 : use t-2, t-1 and t to predict t+1

    :param x: np.ndarray, 1D array of univariate time series values
    :param window: int, number of historical time steps
    :param overlap: int, stride size
    :param num_pred: int, number of future time steps to predict
    :return: X, (N, window) shaped array of historical time steps
             y, (N, num_pred) shaped array of future time steps i.e. labels
    """
    X = np.empty((0, window), dtype='float64')
    y = np.empty((0, num_pred), dtype='float64')

    seq_len = x.shape[0]
    for i in range(0, seq_len - window - num_pred + 1, overlap):
        hist = x[i : i + window]
        pred = x[i + window: i + window + num_pred]

        X = np.append(X, hist.reshape((1, window)), axis=0)
        y = np.append(y, pred.reshape(1, num_pred), axis=0)

    return X, y


def split_dataset(X, y, test_size=0.3, seed=111):
    """
    Splits the dataset into train-validation-test

    :param X: np.ndarray, 2D array of shape N x timesteps
    :param y: np.ndarray, 1D array of shape N x num_pred
    :param test_size: float, proportion of data for val and test sets (split 50/50)
    :param seed: int, seed number for reproducibility
    :return: data_dict, dictionary containing train-val-test sets
    """
    X_train, X2, y_train, y2 = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X2, y2, test_size=0.5, random_state=seed, shuffle=True)

    # Reshaping into 3D inputs for RNN models
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f'X_train shape : {X_train.shape}, y_train shape : {y_train.shape}')
    print(f'X_val shape : {X_val.shape}, y_val shape : {y_val.shape}')
    print(f'X_test shape : {X_test.shape}, y_test shape : {y_test.shape}')

    data_dict = {'train' : [X_train, y_train],
                 'val' : [X_val, y_val],
                 'test' : [X_test, y_test]}

    return data_dict


def normalise(data_dict):
    """
    Min-max scaling of features based on training data
    :param data_dict: dict, dataset split into train-val-test features and labels
    :return: data_dict, with features scaled
             min_val, minimum value of train features
             max_val, maximum value of train features
    """
    X_train = data_dict['train'][0]
    X_val = data_dict['val'][0]
    X_test = data_dict['test'][0]

    min_val, max_val = np.min(X_train), np.max(X_train)
    X_train_scaled = (X_train - min_val) / (max_val - min_val)
    X_val_scaled = (X_val - min_val) / (max_val - min_val)
    X_test_scaled = (X_test - min_val) / (max_val - min_val)

    data_dict['train'][0] = X_train_scaled
    data_dict['val'][0] = X_val_scaled
    data_dict['test'][0] = X_test_scaled

    return data_dict, min_val, max_val


def generate_dataloader(X, y, batch_size=8):
    """
    Generates a PyTorch DataLoader iterator instance
    :param X: np.ndarray, features array
    :param y: np.ndarray, labels array
    :param batch_size: int, batch_size
    :return: loader, PyTorch DataLoader instance
    """
    dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader