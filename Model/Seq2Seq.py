import torch
import torch.nn as nn
from Constants import *


class Encoder(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, in_pad_idx: int, num_layer: int, embed_sz: int):
        super(Encoder, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.num_layer = num_layer
        self.embed = nn.Embedding(input_sz, embed_sz, in_pad_idx)
        self.lstm = nn.LSTM(embed_sz, hidden_sz, num_layers=num_layer)

    def forward(self, input: torch.Tensor, non_padded_len: torch.Tensor, hidden: torch.Tensor = None):
        batch_sz = input.shape[1]
        embedded_input = self.embed(input)
        pps_input = nn.utils.rnn.pack_padded_sequence(embedded_input, non_padded_len, enforce_sorted=False)

        hidden = self.init_hidden(batch_sz)

        _, hidden = self.lstm.forward(pps_input, hidden)

        return hidden

    def init_hidden(self, batch_sz: int):
        return (torch.zeros(self.num_layer, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layer, batch_sz, self.hidden_sz).to(DEVICE))


class Decoder(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, in_pad_idx: int, num_layer: int, embed_sz: int,
                 drop_out: float = 0.1):
        super(Decoder, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.num_layer = num_layer
        self.embed = nn.Embedding(input_sz, embed_sz, in_pad_idx)
        self.lstm = nn.LSTM(embed_sz, hidden_sz, num_layers=num_layer)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(hidden_sz, input_sz)
        self.drop_out = nn.Dropout(drop_out)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        embedded_input = self.embed(input)
        output, hidden = self.lstm(embedded_input, hidden)
        sigmoid_out = self.sigmoid(output)
        fc1_out = self.fc1(sigmoid_out)
        output = self.drop_out(fc1_out)
        probs = self.softmax(output)

        return probs, hidden
