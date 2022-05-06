import torch
from torch import nn

from ResNext.block import ResNextBottleNeck


class ResNext(nn.Module):
    def __init__(self, cardinality, block_width, expansion, num_blocks, num_classes=10):
        super().__init__()

        self.cardinality = cardinality
        self.block_width = block_width
        self.expansion = expansion

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            # 112 맞춰주려고 padding 추가 (흑백이므로 채널1) (224 - 7 + 4) / 2 + 1 = 221 / 2 110 + 1
            nn.Conv2d(in_channels=1, out_channels=self.in_channels, stride=2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, 2) # 3x3 stride 2
        )

        self.conv2 = self._make_layer(num_blocks[0], 1)
        # conv 3, 4, 5만 stride 2 적용한다고 4번째 문단에 나와있음
        self.conv3 = self._make_layer(num_blocks[1], 2)
        self.conv4 = self._make_layer(num_blocks[2], 2)
        self.conv5 = self._make_layer(num_blocks[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # block width는 계속 갱신 됨
        self.fc = nn.Linear(self.cardinality * self.block_width, num_classes)
    
    def _make_layer(self, num_blocks, stride):
        # 채널 수 바뀌는 레이어만 stride 2를 부여함.
        # 기존 논문(ResNet)처럼 아웃풋 - 다시 인풋으로 들어감 (64 - 64 - 256 - 64 - 64 - 256)
        # ResNext에서는 128 - 128 - 256 - 128 - 128 - 256 형식에 2배수임
        # 논문에서는 아웃풋만 신경쓰므로 아웃풋을 맞춰주는 것에 의의가 있음.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNextBottleNeck(self.in_channels, self.cardinality, 
            self.block_width, stride, self.expansion))
            # 마지막 연산 이후 입력 채널은 그룹 넓이 * 블록 채널 * 확장 계수(2배가 원칙)
            self.in_channels = self.cardinality * self.block_width * self.expansion
        # 모든 레이어 작업 이후 block width를 2배 늘려서 다음 채널 사이즈를 작업하게끔 함 (2배 이상) 
        self.block_width *= self.expansion
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) # 채널을 제외하고 flatten
        x = self.fc(x)

        return x


class ResNextMaker(object):
    # signleton
    def __new__(instance):

        # 해당 속성이 없다면
        if not hasattr(instance, "maker"):
            instance.maker = super().__new__(instance)  # 객체 생성 후 바인딩
        return instance.maker
    
    def _makeResNext_101(self):
        """
            return a ResNext 101 Object
        """
        return ResNext(32, 4, 2, [3, 4, 23, 3], 10)