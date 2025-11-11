from PIL import Image
import torch
from torch.utils.data import Dataset
import pywt
import numpy as np
import torch.nn.functional as F
def extract_wavelet_features(image_path, wavelet='haar', level=1):
    image = Image.open(image_path).convert('L')
    image1 = np.array(image)
    """ 使用小波变换提取图像特征 """
    # 执行多级小波分解
    coeffs = pywt.wavedec2(image1, wavelet, level=level)

    # 提取近似系数作为特征
    approx_coeffs = coeffs[0]


    return approx_coeffs
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        wave = extract_wavelet_features(self.images_path[item])
        # print(img)
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
            # print("dataimg",img.size())
        tensor = torch.from_numpy(wave)
        # print("wave.size",tensor.size())


        # 创建两个张量
        tensor1 = img  # 形状为 [3, 48, 48]
        tensor2 = tensor  # 形状为 [24, 24]

        # 调整 tensor2 的形状以匹配 tensor1
        tensor2_resized = F.interpolate(tensor2.unsqueeze(0).unsqueeze(0), size=(48,48), mode='bilinear', align_corners=False).squeeze(0)

        # 将两个张量融合
        fused_tensor = tensor1 + tensor2_resized  # 这里使用逐元素相加作为示例

        # 检查融合后的张量形状
        # print(fused_tensor.shape)  # 应输出: torch.Size([3, 224, 224])

        return fused_tensor.to(torch.float32), label

    @staticmethod
    def collate_fn(batch):

        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
