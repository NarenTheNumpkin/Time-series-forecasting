import torch
from torch import nn

class RecurrentNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, h_n = self.rnn(x)
        x = self.out_proj(output[:, -1, :])
        return x