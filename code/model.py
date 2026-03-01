import torch
from torch import nn

class RecurrentNetwork(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, h_n = self.rnn(x)
        x = self.out_proj(output[:, -1, :])
        return x
class LSTM_GRU(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, model_type="gru"):
        super().__init__()
        if model_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_step = rnn_out[:, -1, :]
        last_step = self.dropout(last_step)
        return self.out_proj(last_step)