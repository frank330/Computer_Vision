"""车牌识别网络结构定义，包含字符识别与颜色识别模型。"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Iterable, Union


def _build_feature_layers(
    cfg: Iterable[Union[int, str]],
    *,
    batch_norm: bool = False,
    first_kernel: int = 5,
    first_padding: Union[int, tuple] = 0,
    default_padding: Union[int, tuple] = 1,
) -> nn.Sequential:
    """根据配置构建卷积 + BN + ReLU 特征提取网络。"""
    layers = []
    in_channels = 3
    for idx, cfg_item in enumerate(cfg):
        if cfg_item == 'M':  # 下采样层
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            continue

        kernel_size = first_kernel if idx == 0 else 3
        padding = first_padding if idx == 0 else default_padding
        conv = nn.Conv2d(in_channels, cfg_item, kernel_size=kernel_size, padding=padding, stride=1)
        block = [conv]
        if batch_norm:
            block.append(nn.BatchNorm2d(cfg_item))
        block.append(nn.ReLU(inplace=True))
        layers.extend(block)
        in_channels = cfg_item
    return nn.Sequential(*layers)


class myNet_ocr(nn.Module):
    """车牌字符识别网络，可选导出模式。"""

    def __init__(self, cfg=None, num_classes: int = 78, export: bool = False):
        super().__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
        self.feature = _build_feature_layers(cfg, batch_norm=True, first_padding=0, default_padding=(1, 1))
        self.export = export
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)

    def forward(self, x):
        """返回字符概率序列，export 模式下输出 argmax 索引。"""
        x = self.feature(x)
        x = self.loc(x)
        x = self.newCnn(x)
        if self.export:
            conv = x.squeeze(2)  # [B, C, W]
            conv = conv.transpose(2, 1)
            return conv.argmax(dim=2)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        conv = x.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        return torch.softmax(conv, dim=2)


myCfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 256]


class myNet(nn.Module):
    """简化版本颜色分类网络（历史遗留，保留兼容性）。"""

    def __init__(self, cfg=None, num_classes: int = 3):
        super().__init__()
        if cfg is None:
            cfg = myCfg
        self.feature = _build_feature_layers(cfg, batch_norm=True, first_padding=0, default_padding=1)
        self.classifier = nn.Linear(cfg[-1], num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(kernel_size=3, stride=1)(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MyNet_color(nn.Module):
    """旧版颜色分类网络，输入为裁剪后的车牌区域。"""

    def __init__(self, class_num: int = 6):
        super().__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(480, 64),
            nn.ReLU(),
            nn.Linear(64, class_num),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.backbone(x)


class myNet_ocr_color(nn.Module):
    """车牌字符 + 颜色联合识别模型。"""

    def __init__(self, cfg=None, num_classes: int = 78, export: bool = False, color_num: int = None):
        super().__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
        self.feature = _build_feature_layers(cfg, batch_norm=True, first_padding=0, default_padding=(1, 1))
        self.export = export
        self.color_num = color_num
        self.conv_out_num = 12
        if color_num:
            self.conv1 = nn.Conv2d(cfg[-1], self.conv_out_num, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(self.conv_out_num)
            self.relu1 = nn.ReLU(inplace=True)
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.color_classifier = nn.Conv2d(self.conv_out_num, color_num, kernel_size=1, stride=1)
            self.color_bn = nn.BatchNorm2d(color_num)
            self.flatten = nn.Flatten()
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)

    def forward(self, x):
        """返回 (字符概率, 颜色向量) 或导出模式下的索引。"""
        x = self.feature(x)
        x_color = None
        if self.color_num:
            x_color = self.conv1(x)
            x_color = self.bn1(x_color)
            x_color = self.relu1(x_color)
            x_color = self.color_classifier(x_color)
            x_color = self.color_bn(x_color)
            x_color = self.gap(x_color)
            x_color = self.flatten(x_color)
        x = self.loc(x)
        x = self.newCnn(x)
        if self.export:
            conv = x.squeeze(2).transpose(2, 1)
            return (conv, x_color) if self.color_num else conv
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        conv = x.squeeze(2).permute(2, 0, 1)
        output = F.log_softmax(conv, dim=2)
        return (output, x_color) if self.color_num else output


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 216)
    model = myNet_ocr(num_classes=78, export=True)
    out = model(x)
    print(out.shape)