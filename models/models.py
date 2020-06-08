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
        self.gru = nn.GRU(1, num_hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        output = output[:, -1, :]
        output = self.fc(output)

        return output


class RecursiveLSTM(nn.Module):
    def __init__(self, num_pred, num_layers=1, num_hidden=50):
        """
        LSTM model for multi-step prediction performed in a recurisve manner, where predicted output for t + 1
        is used as input for prediction of t + 2, and so on (accumulative error)
        :param num_pred: int, number of future time steps to predict
        :param num_layers: number of LSTM layers
        :param num_hidden: int, number of hidden features
        """
        super(RecursiveLSTM, self).__init__()
        self.num_pred = num_pred
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.lstm = nn.LSTM(1, num_hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x):
        pred = torch.empty([x.size()[0], self.num_pred])
        for i in range(self.num_pred):
            output, [h, c] = self.lstm(x)
            output = self.fc(output[:, -1, :])
            pred[:, i] = torch.squeeze(output, -1)

            output = torch.unsqueeze(output, -1)
            x = torch.cat([x, output], 1)[:, 1:, :]

        return pred


class Encoder(nn.Module):
    def __init__(self, num_layers, num_hidden):
        """
        Encoder module for using GRU
        :param num_layers: int, number of LSTM layers
        :param num_hidden: int, number of hidden features
        """
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.gru = nn.GRU(1, num_hidden, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = torch.empty([self.num_layers * 1, x.size()[0], self.hidden_size])
            nn.init.xavier_uniform(hidden_state)

        output, hidden_state = self.gru(x, hidden_state)
        return output, hidden_state


class Decoder(nn.Module):
    def __init__(self, num_layers, num_hidden, num_pred):
        """
        Decoder module for using GRU with teacher forcing
        :param num_layers: int, number of LSTM layers
        :param num_hidden: int, number of hidden features
        """
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.num_pred = num_pred
        self.gru = nn.GRU(num_hidden, num_pred, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = torch.empty([self.num_layers * 1, x.size()[0], self.hidden_size])
            nn.init.xavier_uniform(hidden_state)

        # Initial decoder hidden state will be last hidden state from encoder
        output, _ = self.gru(x, hidden_state)
        # Adjusting output for teacher forcing
        tf = torch.cat(x[:, 0, :], output[:, 1:, :])
        if self.num_pred == 1:
            tf = torch.unsqueeze(tf, -1)
        output, hidden_state = self.gru(tf, hidden_state)

        return output, hidden_state


class EncoderDecoder(nn.Module):
    def __init__(self, num_layers, num_hidden):
        super(EncoderDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        encoder = Encoder(num_layers, num_hidden)
        decoder = Decoder(num_layers, num_hidden)

    def forward(self, x, y):
        enc_output, enc_hidden = encoder(x)
        dec_output, dec_hidden = decoder(y, enc_hidden)


class AttentionLSTM(nn.Module):
    def __init__(self, in_features, query_features, value_features, num_layers=1, num_hidden=50, bidirectional=False):
        """
        Simple LSTM with fully connected layer, with option for bidirectional
        :param num_layers: int, number of LSTM layers
        :param num_hidden: int, number of hidden features
        :param bidirectional: boolean, if True, model is a Bi-LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.c_in, self.c_q, self.c_v = in_features, query_features, value_features

        ### Self attention mechanism
        self.query_W = nn.Linear(self.c_in, self.c_q, bias=False)
        self.key_W = nn.Linear(self.c_in, self.c_q, bias=False)
        self.value_W = nn.Linear(self.c_in, self.c_v, bias=False)
        self.attn_linear = nn.Linear(self.c_v, self.c_in)
        self.lstm = nn.LSTM(self.c_in, num_hidden, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        # self.layer_norm = nn.LayerNorm(self.c_v)

        if bidirectional:
            self.fc = nn.Linear(num_hidden * 2, 1)
        else:
            self.fc = nn.Linear(num_hidden, 1)

    def forward(self, x):
        Q = self.query_W(x)
        K = self.key_W(x)
        V = self.value_W(x)

        dot_product = torch.matmul(Q, K.permute(0, 2, 1)) / (self.c_q ** 0.5)
        scores = torch.softmax(dot_product, dim=2)
        scaled_x = torch.matmul(scores, V) + x
        # scaled_x = self.layer_norm(scaled_x)

        new_x = self.attn_linear(scaled_x) + x
        output, _ = self.lstm(new_x)
        output = output[:, -1, :]
        output = self.fc(output)

        return output