import torch.nn as nn
import torch
from collections import OrderedDict


class CM_GRU(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, kernel_size, stride, padding, innum_layers, batch_size,
                 pock_len):
        super(CM_GRU, self).__init__()
        self.innum_layers = innum_layers
        self.out_channels = out_channels
        self.Conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.MaxPool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.GRU = nn.GRU(pock_len - kernel_size * 2 + 2, out_channels, innum_layers, batch_first=True)


    def forward(self, x):
        h0 = torch.zeros([self.innum_layers, x.size(0), self.out_channels])
        x = self.Conv(x)
        x = self.MaxPool(x)
        device = x.device
        h0 = h0.to(device)
        x = self.GRU(x, h0)
        return x[0]


class Conv1dMax(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.C_M = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.C_M(x)


class CNN_LS(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num=3, hidden_size=64, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.CM_1 = Conv1dMax(in_channels, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.LSTM1 = nn.LSTM(input_size=66 - kernel_size * 2 + 2, hidden_size=out_channels * 3,
                             num_layers=2)
        self.CM_2 = Conv1dMax(hidden_size, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.LSTM2 = nn.LSTM(input_size=out_channels * 3 - kernel_size * 2 + 2, hidden_size=hidden_size, num_layers=2)
        self.Li1 = nn.Linear(hidden_size, int(hidden_size / out_channels))

        self.CNN_LS = nn.Sequential(OrderedDict([('conv_layer0',
                                                  Conv1dMax(in_channels, hidden_size, kernel_size=kernel_size,
                                                            stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.CNN_LS.add_module('ls_layer', nn.LSTM(input_size=hidden_size, hidden_size=hidden_size))
            self.CNN_LS.add_module('conv_layer%d' % (layer_idx + 1),
                                   Conv1dMax(hidden_size, hidden_size, kernel_size=kernel_size + (layer_idx + 1) * 2,
                                             stride=stride,
                                             padding=padding))

        self.CNN_LS.add_module('ls_layer', nn.LSTM(input_size=hidden_size, hidden_size=out_channels))

    def forward(self, x):
        x = self.CM_1(x)
        x = self.LSTM1(x)
        x = self.CM_2(x[0])
        x = self.LSTM2(x)
        x = self.Li1(x[0])
        return x  # batch_size,out_channel,hidden_size/out_channel


class CM_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, kernel_size, stride, padding, innum_layers, batch_size,
                 pock_len):
        super(CM_LSTM, self).__init__()
        self.Conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.MaxPool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.LSTM = nn.LSTM(pock_len - kernel_size * 2 + 2, out_channels, innum_layers, batch_first=True)

    def forward(self, x):
        x = self.Conv(x)
        x = self.MaxPool(x)
        x = self.LSTM(x)
        return x[0]


class Pocket_Encoder(nn.ModuleDict):
    def __init__(self, encoder_type, layer_num, vocab_size, embedding_size, basic_channels, kernel_size,
                 stride, padding, innum_layers, batch_size, pock_len):
        super().__init__()
        self.encoder_type = encoder_type
        self.layer_num = layer_num
        self.embedding_size = embedding_size
        self.basic_channels = basic_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.innum_layers = innum_layers
        self.batch_size = batch_size
        self.pock_len = pock_len

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.fc = nn.Linear(basic_channels, 2)
        self.encoder = self.makelayers()

    def makelayers(self):
        layers = []
        if self.encoder_type == 'GRU':
            layers.append(CM_GRU(self.embedding_size, self.basic_channels * self.layer_num,
                                 self.basic_channels * (self.layer_num - 1), self.kernel_size, self.stride,
                                 self.padding, self.innum_layers, self.batch_size, self.pock_len))
            for i in range(self.layer_num - 2):
                layers.append(
                    CM_GRU(self.basic_channels * (self.layer_num - i), self.basic_channels * (self.layer_num - i - 1),
                           self.basic_channels * (self.layer_num - i - 2), self.kernel_size, self.stride,
                           self.padding, self.innum_layers, self.batch_size,
                           self.basic_channels * (self.layer_num - i - 1)))
            layers.append(
                CM_GRU(self.basic_channels * 2, self.basic_channels, self.basic_channels, self.kernel_size, self.stride,
                       self.padding, self.innum_layers, self.batch_size, self.basic_channels))
        elif self.encoder_type == 'LSTM2':
            layers.append(CM_LSTM(self.embedding_size, self.basic_channels * self.layer_num,
                                  self.basic_channels * (self.layer_num - 1), self.kernel_size, self.stride,
                                  self.padding, self.innum_layers, self.batch_size, self.pock_len))
            for i in range(self.layer_num - 2):
                layers.append(
                    CM_LSTM(self.basic_channels * (self.layer_num - i), self.basic_channels * (self.layer_num - i - 1),
                            self.basic_channels * (self.layer_num - i - 2), self.kernel_size, self.stride,
                            self.padding, self.innum_layers, self.batch_size,
                            self.basic_channels * (self.layer_num - i - 1)))
            layers.append(
                CM_LSTM(self.basic_channels * 2, self.basic_channels, self.basic_channels, self.kernel_size,
                        self.stride,
                        self.padding, self.innum_layers, self.batch_size, self.basic_channels))
        elif self.encoder_type == 'LSTM1':
            layers.append(
                CNN_LS(self.embedding_size, out_channels=32))
        return nn.ModuleList(layers)

    def forward(self, data):
        x = self.embedding(data)
        x = x.permute(0, 2, 1)
        for layer in self.encoder:
            x = layer(x)
        #使用LSTM1时注意注释下一行
        x = self.fc(x)
        return x
