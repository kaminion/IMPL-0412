import torch
from torch import nn

from ResNext.block import ResNextBottleNeck


class ResNext(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            # 112 맞춰주려고 padding 추가 (흑백이므로 채널1) (224 - 7 + 6) / 2 + 1 = 223 / 2 => 111 + 1 = 112
            nn.Conv2d(in_channels=1, out_channels=self.in_channels, stride=2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 3x3 stride 2 112 - 3 + 2 = 111 / 2 = 55 + 1 = 56
        )

        self.conv2 = self._make_layer(num_blocks[0], 64, 1)
        # conv 3, 4, 5만 stride 2 적용한다고 4번째 문단에 나와있음
        self.conv3 = self._make_layer(num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(num_blocks[3], 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # block width는 계속 갱신 됨
        self.fc = nn.Linear(512 * ResNextBottleNeck.expansion, num_classes)
    
    def _make_layer(self, num_blocks, out_channels, stride):
        # 채널 수 바뀌는 레이어만 stride 2를 부여함.
        # 기존 논문(ResNet)처럼 아웃풋 - 다시 인풋으로 들어감 (64 - 64 - 256 - 64 - 64 - 256)
        # 논문에서는 아웃풋만 신경쓰므로 아웃풋을 맞춰주는 것에 의의가 있음.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # output을 맨 처음에 64를 주었으나, 그룹 컨볼루션에 의해 4 * 32 = 128채널이 아웃풋으로 나옴
            # 그래서 결론적으로는 64 * 4 = 256이 아웃풋으로 나오는게 맞음
            layers.append(ResNextBottleNeck(self.in_channels, out_channels, stride))
            # 기본 Residual block과 동일하다.
            self.in_channels = out_channels * ResNextBottleNeck.expansion
        # 모든 레이어 작업 이후 block width를 2배 늘려서 다음 채널 사이즈를 작업하게끔 함 (2배 이상) 
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
    
    def _makeResNext_34(self):
        """
            return a ResNext 34 Object
        """
        return ResNext([2, 4, 6, 3])

    def _makeResNext_50(self):
        """
            return a ResNext 50 Object
        """
        return ResNext([3, 4, 6, 3])

    def _makeResNext_101(self):
        """
            return a ResNext 101 Object
        """
        return ResNext([3, 4, 23, 3])