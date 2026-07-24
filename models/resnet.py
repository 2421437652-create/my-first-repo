# ============================================
# models/resnet.py
# ============================================
import torch
import torch.nn as nn
from torchvision.models import resnet18



class SplitResNet18(nn.Module):


    def __init__(
            self,
            num_classes=10):

        super().__init__()


        model = resnet18(
            weights=None
        )


        self.conv1 = model.conv1

        self.bn1 = model.bn1

        self.relu = model.relu

        self.maxpool = model.maxpool


        self.layer1 = model.layer1

        self.layer2 = model.layer2

        self.layer3 = model.layer3

        self.layer4 = model.layer4


        self.avgpool = model.avgpool


        self.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )


    def forward(self,x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)


        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)


        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

        return x