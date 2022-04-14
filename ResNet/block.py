import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    3*3 Convolution Layer 2개로 이루어진 Block이다.
    """

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        # 원본 사이즈 보전 padding = 1
        # 3x3 Conv가 2개 들어있다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        # 연산용
        self.shortcut = nn.Sequential()

        self.ReLU = F.relu()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.ReLU(x)
        return x
