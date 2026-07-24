import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class ResNet18_CIFAR100(ResNet):

    def __init__(self):

        super().__init__(
            block=BasicBlock,
            layers=[2,2,2,2],
            num_classes=100
        )


        # 修改第一层
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        # CIFAR不需要maxpool
        self.maxpool = nn.Identity()