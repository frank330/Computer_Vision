"""车牌颜色识别辅助模块 | Utilities for license plate color classification."""
import warnings
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from plate_recognition.plateNet import MyNet_color


class MyNet(nn.Module):
    """简单的颜色分类网络结构；Lightweight color classification network."""

    def __init__(self, class_num=6):
        super(MyNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),  # 0
            torch.nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0),
            nn.Flatten(),
            nn.Linear(480, 64),
            nn.Dropout(0),
            nn.ReLU(),
            nn.Linear(64, class_num),
            nn.Dropout(0),
            nn.Softmax(1)
        )

    def forward(self, x):
        logits = self.backbone(x)
        return logits


def init_color_model(model_path, device):
    """加载车牌颜色识别模型权重；Load plate color classification model weights."""

    class_num = 6
    warnings.filterwarnings('ignore')
    net = MyNet_color(class_num)
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    net.eval().to(device)
    modelc = net
    return modelc


def plate_color_rec(img, model, device):
    """根据输入车牌图片预测颜色类别；Predict plate color from cropped plate image."""
    class_name = ['黑色', '蓝色', '', '绿色', '白色', '黄色']
    data_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(data_input, (34, 9))
    image = np.transpose(image, (2, 0, 1))
    img = image / 255
    img = torch.tensor(img)

    normalize = transforms.Normalize(mean=[0.4243, 0.4947, 0.434],
                                     std=[0.2569, 0.2478, 0.2174])
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0).to(device).float()
    xx = model(img)

    return class_name[int(torch.argmax(xx, dim=1)[0])]

