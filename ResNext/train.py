import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from ResNext.resnext import ResNextMaker
from ResNext.utils import get_param_train, train_val

from global_utils import create_directory, history_to_JSON  # load to global_utils

torch.manual_seed(233)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.DataParallel(ResNextMaker()._makeResNext_101()).to(device)

# 데이터 다운을 위한 directory 생성
path2data = './data'
create_directory(path2data)


# Data Augmentation
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

# reduction: sum 하면서 하나의 스칼라로 리턴함.
# 안하면 각 배치사이즈나 데이터별로 따로 계산함(default: mean)
loss_func = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-03, weight_decay=0.0001)  # 0.001

# 러닝레이트 줄여주는 스케쥴러, factor는 감소 시키는 비율, patience는 향상이 안될 때 얼마나 참을 건지(epochs)
# mode는 모니터링 되는 값이 최소가 되어야하는지 최대가 되어야하는지, val_acc(정확도)일 경우 최대, val_loss(loss값)일 경우 작아야 좋으므로 최소가 되어야함
# 여기서는 loss를 측정하므로 mode='min'
lr_scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10)

hyper_param = get_param_train(optimizer, loss_func, train_dl, val_dl, lr_scheduler, device, False)

model, loss_hist, metric_hist = train_val(model, hyper_param)

history_to_JSON("ResNext", loss_hist)