"""
自定义数据增强变换模块
用于图像分割任务的数据预处理，支持图像和标签的同步变换
"""
import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    """
    如果图像的最小边长小于指定尺寸，则进行填充
    
    Args:
        img: PIL Image对象
        size: 目标最小尺寸
        fill: 填充值，默认为0
        
    Returns:
        PIL Image: 填充后的图像
    """
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        # 计算需要填充的高度和宽度
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        # F.pad参数：(left, top, right, bottom)
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    """
    组合多个变换操作，按顺序应用到图像和标签上
    """
    
    def __init__(self, transforms):
        """
        Args:
            transforms: 变换操作列表
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        依次应用所有变换
        
        Args:
            image: 输入图像
            target: 标签图像
            
        Returns:
            tuple: (变换后的图像, 变换后的标签)
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    """
    随机调整图像和标签的大小
    用于数据增强，增加模型的泛化能力
    """
    
    def __init__(self, min_size, max_size=None):
        """
        Args:
            min_size: 最小尺寸
            max_size: 最大尺寸，如果为None则等于min_size
        """
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        """
        随机选择一个尺寸并调整图像和标签大小
        
        Args:
            image: 输入图像
            target: 标签图像
            
        Returns:
            tuple: (调整后的图像, 调整后的标签)
        """
        # 随机选择目标尺寸
        size = random.randint(self.min_size, self.max_size)
        
        # 将图像的最小边长缩放到size大小（保持宽高比）
        image = F.resize(image, size)
        
        # 标签使用最近邻插值，避免产生新的类别值
        # 注意：torchvision 0.9.0以后使用InterpolationMode.NEAREST
        # 之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    """
    随机水平翻转图像和标签
    用于数据增强
    """
    
    def __init__(self, flip_prob):
        """
        Args:
            flip_prob: 翻转概率，范围[0, 1]
        """
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        """
        根据概率决定是否进行水平翻转
        
        Args:
            image: 输入图像
            target: 标签图像
            
        Returns:
            tuple: (翻转后的图像, 翻转后的标签)
        """
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    """
    随机垂直翻转图像和标签
    用于数据增强
    """
    
    def __init__(self, flip_prob):
        """
        Args:
            flip_prob: 翻转概率，范围[0, 1]
        """
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        """
        根据概率决定是否进行垂直翻转
        
        Args:
            image: 输入图像
            target: 标签图像
            
        Returns:
            tuple: (翻转后的图像, 翻转后的标签)
        """
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    """
    随机裁剪图像和标签到指定尺寸
    用于数据增强和统一输入尺寸
    """
    
    def __init__(self, size):
        """
        Args:
            size: 裁剪后的尺寸（正方形）
        """
        self.size = size

    def __call__(self, image, target):
        """
        随机裁剪图像和标签
        
        Args:
            image: 输入图像
            target: 标签图像
            
        Returns:
            tuple: (裁剪后的图像, 裁剪后的标签)
        """
        # 如果图像小于目标尺寸，先进行填充
        image = pad_if_smaller(image, self.size)
        # 标签填充使用255（通常表示背景或忽略区域）
        target = pad_if_smaller(target, self.size, fill=255)
        
        # 获取随机裁剪参数
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        
        # 应用相同的裁剪参数到图像和标签
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class ToTensor(object):
    """
    将PIL Image转换为PyTorch Tensor
    """
    
    def __call__(self, image, target):
        """
        转换图像和标签为tensor格式
        
        Args:
            image: PIL Image，转换为[0,1]范围的tensor
            target: PIL Image，转换为int64类型的tensor（二值化处理）
            
        Returns:
            tuple: (图像tensor, 标签tensor)
        """
        # 图像转换为tensor，自动归一化到[0,1]范围
        image = F.to_tensor(image)
        # 标签转换为numpy数组
        target_array = np.array(target)
        # 二值化处理：将标签值转换为0或1
        # 0: 背景
        # 1: 前景
        # 255: 忽略区域（保持不变，用于padding）
        target_array = np.where(target_array == 255, 255, np.where(target_array > 0, 1, 0))
        # 转换为int64类型的tensor
        target = torch.as_tensor(target_array, dtype=torch.int64)
        return image, target


class Normalize(object):
    """
    对图像进行标准化（归一化）
    使用均值和标准差进行归一化，有助于模型训练稳定
    """
    
    def __init__(self, mean, std):
        """
        Args:
            mean: 各通道的均值
            std: 各通道的标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """
        对图像进行标准化，标签保持不变
        
        Args:
            image: 图像tensor
            target: 标签tensor（不进行归一化）
            
        Returns:
            tuple: (标准化后的图像, 标签)
        """
        # 只对图像进行归一化，标签保持不变
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
