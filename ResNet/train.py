import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary


from resnet import ResNetMaker

from global_utils import get_lr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetMaker()._makeResNet_152().to(device)

summary(model, (3, 224, 224), device=device.type)

# reduction: sum 하면서 하나의 스칼라로 리턴함.
# 안하면 각 배치사이즈나 데이터별로 따로 계산함(default: mean)
loss_function = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=1e-04)  # 0.0001

# 러닝레이트 줄여주는 스케쥴러, factor는 감소 시키는 비율, patience는 향상이 안될 때 얼마나 참을 건지(epochs)
# mode는 모니터링 되는 값이 최소가 되어야하는지 최대가 되어야하는지, val_acc(정확도)일 경우 최대, val_loss(loss값)일 경우 작아야 좋으므로 최소가 되어야함
# 여기서는 loss를 측정하므로 mode='min'
lr_scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10)
