from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

# 이전과 동일하게 Residual Block을 사용한다 함
class ResNextBottleNeck(nn.Module):
    """
        description:
        이전과 동일하게 residual block 사용
        1. 같은 하이퍼파라미터 공유(width, filter-size)
        2. 아웃풋 사이즈는 1/2, 채널 사이즈는 2배
        input_channle과 output channel이 4차원임 (그룹은 32)
        이걸로 인풋 / 아웃풋이 결정됨
    """
    def __init__(self, in_channels, cardinality=32, block_width=4, stride=1, expansion=2):
        super().__init__()

        group_channels = cardinality * block_width
        out_channels = group_channels * expansion

        # 블럭 C 그대로 구현
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, group_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(group_channels),
            nn.ReLU(),
            # 각 스테이지의 3x3 컨볼루션 레이어만 stride=2
            nn.Conv2d(group_channels, group_channels, kernel_size=3, stride=stride, bias=False, groups=cardinality),
            nn.BatchNorm2d(group_channels),
            nn.ReLU(),
            nn.Conv2d(group_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()
        self.ReLU = nn.ReLU()

        # 차원이 안맞을 경우 차원 조절
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 논문에서 말하는 Eqn.2 는 1x1 Conv, stride 적용만 해서 차원만 맞춤
                nn.Conv2d(in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False),
                # 모든 Conv 연산 후에는 Batch를 적용한다 했음
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.ReLU(x)

        return x
