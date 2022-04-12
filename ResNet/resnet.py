import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from block import BasicBlock


class PracticeResNet(nn.Module):
    # block: , num_block: 각 모듈의 갯수 (논문에서 구조 부분 x2, x2 로 표시되어있는 부분)
    def __init__(self, block, num_block, num_classes=10,
                 init_weights=True):

        super(PracticeResNet, self).__init__()

        # 초기 채널은 64 채널부터 시작함
        self.in_channels = 64

        # 논문에 의하면 매 Conv2D마다 BatchNorm을 함
        # 112 사이즈로 맞추므로 패딩 사이즈 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      padding=3, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2_x
        self.conv3_x
        self.conv4_x
        self.conv5_x

    # block: block instance
    # num_block은 모듈 구조의 갯수이다.
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        return nn.Sequential(*layers)

    def forward(self, x):
        return x
