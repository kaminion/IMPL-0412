import torch
from resnet import ResNetMaker
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetMaker()._makeResNet_152().to(device)

summary(model, (3, 224, 224), device=device.type)
