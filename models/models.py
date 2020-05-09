import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, num_layers, num_nodes):
        """
        Simple feed-forward neural network
        :param num_layers: int, number of layers of the network
        :param num_nodes: list, number of nodes for each layer, should be of length num_layers + 1
        """
        super(ANN, self).__init__()
        self.fc_list = nn.ModuleList()
        for layer in range(num_layers - 1):
            self.fc_list.append(nn.Linear(num_nodes[layer], num_nodes[layer + 1]))
        self.fc = nn.Linear(num_nodes[-2], num_nodes[-1])

    def forward(self, x):
        x = self.flatten(x)
        for l in self.fc_list:
            x = F.relu(l(x))
        x = self.fc(x)
        return x

    def flatten(self, x):
        return torch.squeeze(x, 2)


class LSTM(nn.Module):
    def __init__(self, num_layers=1, num_hidden=50, bidirectional=False):
        """
        Simple LSTM with fully connected layer, with option for bidirectional
        :param num_layers: int, number of LSTM layers
        :param num_hidden: int, number of hidden features
        :param bidirectional: boolean, if True, model is a Bi-LSTM
        """
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.lstm = nn.LSTM(1, num_hidden, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(num_hidden * 2, 1)
        else:
            self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc(output)

        return output


class GRU(nn.Module):
    def __init__(self, num_layers=1, num_hidden=50):
        """
        Simple GRU with fully connected layer
        :param num_layers: int, number of LSTM layers
        :param num_hidden: int, number of hidden features
        """
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.lstm = nn.LSTM(1, num_hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc(output)

        return output