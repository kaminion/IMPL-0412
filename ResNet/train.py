import os
import sys  # load to global_utils
sys.path.append(os.getcwd())

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, utils
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from global_utils import create_directory, get_lr, history_to_JSON
from utils import get_param_train, train_val
from resnet import ResNetMaker

torch.manual_seed(233)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Multi GPU Setting
model = nn.DataParallel(ResNetMaker()._makeResNet_101()).to(device)

# 데이터 다운을 위한 directory 생성
path2data = './data'
create_directory(path2data)

# Data Augmentation
# Gray Scale만 가지므로 Normalize는 1차원만 설정한다.
train_transformation = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(), # 0-1 scaling
    transforms.Normalize((0.5,), (0.5,)),
])

val_transformation = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.FashionMNIST(
    path2data, download=True, train=True, transform=train_transformation)
val_ds = datasets.FashionMNIST(
    path2data, download=True, train=False, transform=val_transformation)


# DataLoader에 적재
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=True)


# 이미지 확인
# img_np = utils.make_grid([train_ds[0][0], train_ds[1][0], train_ds[2][0]], 4, 2).numpy()
# img_np_tr = np.transpose(img_np, (1, 2, 0))
# plt.imshow(img_np_tr)
# plt.show()

# summary(model, (1, 224, 224), device=device.type)

# reduction: sum 하면서 하나의 스칼라로 리턴함.
# 안하면 각 배치사이즈나 데이터별로 따로 계산함(default: mean)
loss_function = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=1e-03, weight_decay=0.0001)  # 0.001

# 러닝레이트 줄여주는 스케쥴러, factor는 감소 시키는 비율, patience는 향상이 안될 때 얼마나 참을 건지(epochs)
# mode는 모니터링 되는 값이 최소가 되어야하는지 최대가 되어야하는지, val_acc(정확도)일 경우 최대, val_loss(loss값)일 경우 작아야 좋으므로 최소가 되어야함
# 여기서는 loss를 측정하므로 mode='min'
lr_scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10)

hyper_param = get_param_train(optimizer, loss_function, train_dl, val_dl, lr_scheduler, device, False)

model, loss_hist, metric_hist = train_val(model, hyper_param)


history_to_JSON("ResNet", metric_hist)