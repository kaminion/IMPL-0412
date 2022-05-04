from torch import nn


class ResNext(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()


class ResNextMaker(object):
    # signleton
    def __new__(instance):

        # 해당 속성이 없다면
        if not hasattr(instance, "maker"):
            instance.maker = super().__new__(instance)  # 객체 생성 후 바인딩
        return instance.maker