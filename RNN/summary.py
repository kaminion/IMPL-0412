import torch 
from torch import nn
from torchsummary import summary

model = nn.RNN(hidden_size=5, input_size=5)
summary(model, (1, 2, 5))