import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    3*3 Convolution Layer 2개로 이루어진 Block이다.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 원본 사이즈 보전 padding = 1
        # 3x3 Conv가 2개 들어있다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # 나갈때 차원보전해야하므로 패딩값과 stride 조정
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # eq1의 경우
        self.shortcut = nn.Sequential()

        # 차원이 바뀌거나, input/output 채널이 바뀔 경우 (선형 곱으로 차원을 맞춰줌, eq(1) case가 아닌 경우)
        # 이 경우 stride 2가 적용되어 차원이 바뀔 때, in_channels, out_channels
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 논문에서 말하는 W는 1x1 Conv, stride 적용만 해서 차원만 맞춤
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                # 모든 Conv 연산 후에는 Batch를 적용한다 했음
                nn.BatchNorm2d(out_channels)
            )

        self.ReLU = F.relu()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.ReLU(x)
        return x
