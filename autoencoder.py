# Import Module
import os
import numpy as np

# Import PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as torch_utils

# Import Custom Module

# Implementation from article : SAE network : a deep learning method for traffic flow prediction
# (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8572451)
# Reference code : https://github.com/xiaochus/TrafficFlowPrediction, https://github.com/ShayanPersonal/stacked-autoencoder-pytorch

# Stacked AutoEncoder
class SAE(nn.Module):
    def __init__(self, layers, dropout): # layer = [12, 300, 300, 300, 12]
        super(SAE, self).__init__()
        self.ae1 = _AE(layers[0], layers[1], dropout)
        self.ae2 = _AE(layers[1], layers[2], dropout)
        self.ae3 = _AE(layers[2], layers[3], dropout)
        self.predict = nn.Linear(layers[3], layers[4])

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        y = self.predict(a3)

        return y


# three-layer AutoEncoder : building block
class _AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x):
        #x = x.detach().cpu()
        h = self.encoder(x)
        h = self.dropout(h)
        
        if self.training:
            x_reconstruct = self.decoder(h)
            loss = self.criterion(x_reconstruct, x)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        return h


# Reference code : https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
# 수정해야함
# Deep AutoEncoder 12 layer
class DAE12(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(True),
            nn.Linear(10, 8),
            nn.ReLU(True),
            nn.Linear(8, 6),
            nn.ReLU(True),
            nn.Linear(6, 4),
            nn.ReLU(True),
            nn.Linear(4, 2),
            nn.ReLU(True),
            nn.Linear(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(True),
            nn.Linear(2, 4),
            nn.ReLU(True),
            nn.Linear(4, 6),
            nn.ReLU(True),
            nn.Linear(6, 8),
            nn.ReLU(True),
            nn.Linear(8, 10),
            nn.ReLU(True),
            nn.Linear(10, 12),
        )
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x
