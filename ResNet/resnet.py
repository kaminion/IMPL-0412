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

        # 초기 채널은 64 채널부터 시작함 (논문 구조 참조)
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

        # channel수는 64, 128, 256, 512순이다. 증가한다.
        self.conv2_x = self._make_layer(
            block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(
            block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(
            block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(
            block, 512, num_block[3], 2)

        # 아웃풋 사이즈만 지정해주면 되는 Pooling Operation을 사용
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 마지막 채널을 받을 수 있는 차원을 만듬
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # block: block instance
    # num_block은 모듈 구조의 갯수이다. (논문 구조 참고 x3, x4, x6, x3 과 같은 구조)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
            resnet layer를 만드는 method, residual block을 resnet layer라고 한다. 

            Args:
                block: block.py에서 정의한 block type, basic block 혹은 bottle neck block이다.
                out_channels: 만들어질 레이어의 output channels
                num_blocks: 레이어 별 포함되는 블록 수
                stride: 만들어질 레이어의 첫번째 stride

            Return: 
                residual layer가 반환된다.
        """

        # 레이어마다 포함되는 block 수가 num_block 이다.
        # 첫번째 블록은 stride를 2로, 나머지 블록은 모두 stride를 1로 유지함 (사이즈를 줄이는 역할은 첫번째 에서 담당)
        # list + list = list, 첫번째 블록을 제외한 1짜리 블록을 곱하기로 여러개를 만듬
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # 배열에 residual block 추가
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # 차원 변경됨 (여기서 BottleNeck 같은 경우는 4배가 됨)
            self.in_channels = out_channels * block.expansion

        # Unpacking Container Type
        return nn.Sequential(*layers)

    def forward(self, x):

        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        # avg_pool로 1 * 1 만들었으니 flatten이 필요 (FC 레이어는 1차원임)
        # 행, 렬 순으로 변환 (그러나 Pytorch의 Tensor Convention은 batch_size, channel, width, height 순이므로 채널 사이즈 외 다 flatten)
        output = output.view(output.size[0], -1)
        output = self.fc(output)

        return output
