"""车辆颜色分类网络定义。"""

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["myNet", "myResNet18"]

# 默认卷积层配置：数值为输出通道，'M' 代表 3x3 最大池化
_DEFAULT_CFG = [32, "M", 64, "M", 96, "M", 128, "M", 256]


def _build_feature_layers(cfg, batch_norm: bool = True) -> nn.Sequential:
    """根据配置构建卷积堆叠，支持可选 BN。"""
    layers = []
    in_channels = 3
    for idx, item in enumerate(cfg):
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            continue
        kernel = 5 if idx == 0 else 3
        padding = 0 if idx == 0 else 1
        conv = nn.Conv2d(in_channels, item, kernel_size=kernel, stride=1, padding=padding)
        block = [conv]
        if batch_norm:
            block.append(nn.BatchNorm2d(item))
        block.append(nn.ReLU(inplace=True))
        layers.extend(block)
        in_channels = item
    return nn.Sequential(*layers)


class myNet(nn.Module):
    """轻量级车辆颜色分类网络。"""

    def __init__(self, cfg=None, num_classes: int = 3):
        super().__init__()
        if cfg is None:
            cfg = _DEFAULT_CFG
        self.feature = _build_feature_layers(cfg)
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1)
        self.classifier = nn.Linear(cfg[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class myResNet18(nn.Module):
    """基于 ResNet18 的颜色分类网络（可快速迁移学习）。"""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        backbone.averagePool = nn.AvgPool2d((5, 5), stride=1, ceil_mode=True)
        self.backbone = backbone
        self.cls = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.averagePool(x)
        x = x.view(x.size(0), -1)
        return self.cls(x)


if __name__ == "__main__":
    dummy = torch.randn(1, 3, 64, 64)
    net = myNet(num_classes=3)
    print(net(dummy).shape)