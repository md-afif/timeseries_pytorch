import torch

def MAE(y_true, y_pred):
    """
    Calculates MAE for each predicted time step
    :param y_true: torch.Tensor, tensor of ground truth values
    :param y_pred: torch.Tensor, tensor of predicted values
    :return: torch.Tensor
    """
    return torch.mean(torch.abs(y_true - y_pred), dim=0)


def MSE(y_true, y_pred):
    """
    Calculates MSE for each predicted time step
    :param y_true: torch.Tensor, tensor of ground truth values
    :param y_pred: torch.Tensor, tensor of predicted values
    :return: torch.Tensor
    """
    return torch.mean(torch.pow((y_true - y_pred), 2), dim=0)


def MAPE(y_true, y_pred):
    """
    Calculates MAPE for each predicted time step
    :param y_true: torch.Tensor, tensor of ground truth values
    :param y_pred: torch.Tensor, tensor of predicted values
    :return: torch.Tensor
    """
    return torch.mean(torch.abs(y_true - y_pred / y_true), dim=0)


def bias(y_true, y_pred):
    """
    Calculates bias for each predicted time step
    :param y_true: torch.Tensor, tensor of ground truth values
    :param y_pred: torch.Tensor, tensor of predicted values
    :return: torch.Tensor
    """
    return torch.mean(y_true - y_pred, dim=0)