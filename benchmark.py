import torch
from ptflops import get_model_complexity_info # pip install ptflops
import sys, os
from torchsummary import summary
from torchvision.models import resnext101_32x8d


sys.path.append(os.getcwd() + os.altsep + 'ResNet')
sys.path.append(os.getcwd() + os.altsep + 'ResNext')

# print(os.getcwd() + os.altsep + 'ResNext')
from ResNet.resnet import ResNetMaker
from ResNext.resnext import ResNextMaker

# model = ResNetMaker()._makeResNet_101()
model = ResNextMaker()._makeResNext_101()
# model = resnext101_32x8d()
macs, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
print(f"FLOPS:: {macs}\n")
print(f"params:: {params}")
# summary(model, (1, 224, 224))
