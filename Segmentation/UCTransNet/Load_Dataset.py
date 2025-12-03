# -*- coding: utf-8 -*-
"""
数据集加载模块
该模块提供了用于图像分割任务的数据加载、预处理和数据增强功能。
主要包括：
1. 数据增强函数（随机旋转、翻转等）
2. 数据生成器类（训练集和验证集）
3. 图像到图像的数据集类
"""

import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage

def random_rot_flip(image, label):
    """
    对图像和标签进行随机旋转和翻转的数据增强
    
    功能：
        - 随机旋转图像和标签（0度、90度、180度或270度）
        - 随机沿水平或垂直轴翻转图像和标签
    
    Args:
        image: 输入图像（numpy数组或PIL图像）
        label: 对应的标签/掩码（numpy数组或PIL图像）
    
    Returns:
        tuple: (增强后的图像, 增强后的标签)
    """
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    """
    对图像和标签进行随机角度旋转的数据增强
    
    功能：
        - 在 -20 到 20 度之间随机选择一个角度
        - 对图像和标签进行相同角度的旋转
    
    Args:
        image: 输入图像（numpy数组）
        label: 对应的标签/掩码（numpy数组）
    
    Returns:
        tuple: (旋转后的图像, 旋转后的标签)
    """
    angle = np.random.randint(-20, 20)  # 随机选择旋转角度（-20到20度）
    # order=0 表示使用最近邻插值（适合标签），reshape=False 保持原始尺寸
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    """
    训练集数据增强生成器
    
    功能：
        - 对训练数据进行随机增强（旋转、翻转）
        - 将图像和标签调整到指定尺寸
        - 转换为PyTorch张量格式
    
    使用场景：
        用于训练阶段的数据增强，提高模型的泛化能力
    """
    def __init__(self, output_size):
        """
        初始化随机生成器
        
        Args:
            output_size: 输出图像的目标尺寸，格式为 [height, width] 或 (height, width)
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        对样本进行数据增强处理
        
        Args:
            sample: 包含 'image' 和 'label' 键的字典
        
        Returns:
            dict: 处理后的样本字典，包含增强后的 'image' 和 'label'
        """
        image, label = sample['image'], sample['label']
        # 转换为PIL图像格式以便进行数据增强
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size  # 获取原始图像尺寸
        
        # 随机应用数据增强（50%概率进行旋转翻转，25%概率进行角度旋转）
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)

        # 如果图像尺寸与目标尺寸不一致，则进行缩放
        if x != self.output_size[0] or y != self.output_size[1]:
            # order=3 表示使用三次样条插值（适合图像），order=0 表示最近邻插值（适合标签）
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为PyTorch张量格式
        image = F.to_tensor(image)  # 图像转换为 [C, H, W] 格式的浮点张量
        label = to_long_tensor(label)  # 标签转换为长整型张量
        sample = {'image': image, 'label': label}
        return sample

class ValGenerator(object):
    """
    验证集数据生成器
    
    功能：
        - 对验证数据进行预处理（不进行随机增强）
        - 将图像和标签调整到指定尺寸
        - 转换为PyTorch张量格式
    
    使用场景：
        用于验证阶段的数据处理，保持数据的一致性以便准确评估模型性能
    """
    def __init__(self, output_size):
        """
        初始化验证生成器
        
        Args:
            output_size: 输出图像的目标尺寸，格式为 [height, width] 或 (height, width)
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        对样本进行预处理（不进行随机增强）
        
        Args:
            sample: 包含 'image' 和 'label' 键的字典
        
        Returns:
            dict: 处理后的样本字典，包含预处理后的 'image' 和 'label'
        """
        image, label = sample['image'], sample['label']
        # 转换为PIL图像格式
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size  # 获取原始图像尺寸
        
        # 如果图像尺寸与目标尺寸不一致，则进行缩放（验证集不进行随机增强）
        if x != self.output_size[0] or y != self.output_size[1]:
            # order=3 表示使用三次样条插值（适合图像），order=0 表示最近邻插值（适合标签）
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为PyTorch张量格式
        image = F.to_tensor(image)  # 图像转换为 [C, H, W] 格式的浮点张量
        label = to_long_tensor(label)  # 标签转换为长整型张量
        sample = {'image': image, 'label': label}
        return sample

def to_long_tensor(pic):
    """
    将图像转换为PyTorch长整型张量
    
    功能：
        - 将PIL图像或numpy数组转换为PyTorch张量
        - 转换为长整型（int64），适合用作标签/掩码
    
    Args:
        pic: PIL图像或numpy数组
    
    Returns:
        torch.Tensor: 长整型张量
    """
    # 处理numpy数组：将图像转换为uint8类型的numpy数组
    img = torch.from_numpy(np.array(pic, np.uint8))
    # 转换为长整型（向后兼容性考虑）
    return img.long()

def correct_dims(*images):
    """
    修正图像的维度
    
    功能：
        - 如果图像是2维的（灰度图），则在第3维添加一个通道维度
        - 确保所有图像都有正确的维度格式
    
    Args:
        *images: 可变数量的图像数组（numpy数组）
    
    Returns:
        如果只有一个图像，返回修正后的单个图像
        如果有多个图像，返回修正后的图像列表
    """
    corr_images = []
    for img in images:
        # 如果是2维图像（缺少通道维度），则添加一个通道维度
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    # 如果只有一个图像，直接返回；否则返回列表
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class ImageToImage2D(Dataset):
    """
    图像到图像的数据集类（用于图像分割任务）
    
    功能：
        - 从指定路径加载图像和对应的标签/掩码
        - 对图像和标签进行预处理（调整尺寸、归一化等）
        - 支持数据增强（通过joint_transform参数）
        - 支持one-hot编码的标签格式
    

    
    Args:
        dataset_path (str): 数据集根目录路径
        joint_transform (Callable, optional): 数据增强变换函数，如 RandomGenerator 或 ValGenerator
            如果为 None，则使用默认的 ToTensor 转换
        one_hot_mask (int, optional): 是否将标签转换为one-hot编码格式
            如果 > 0，则转换为one-hot编码，数值表示类别数
        image_size (int, optional): 图像的目标尺寸，默认为224
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =224) -> None:
        """
        初始化数据集
        
        Args:
            dataset_path: 数据集根目录路径
            joint_transform: 数据增强变换函数（如 RandomGenerator 或 ValGenerator）
            one_hot_mask: 是否使用one-hot编码的标签格式
            image_size: 图像的目标尺寸（宽度和高度）
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        # 设置输入图像和标签的路径
        self.input_path = os.path.join(dataset_path, 'images')  # 输入图像目录
        self.output_path = os.path.join(dataset_path, 'labels')  # 标签/掩码目录
        self.images_list = os.listdir(self.input_path)  # 获取所有图像文件名列表
        self.one_hot_mask = one_hot_mask  # 是否使用one-hot编码

        # 设置数据变换函数
        if joint_transform:
            self.joint_transform = joint_transform  # 使用提供的数据增强函数
        else:
            # 如果没有提供，则使用默认的ToTensor转换
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        """
        返回数据集的大小（图像数量）
        
        Returns:
            int: 数据集中图像的数量
        """
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
        
        Returns:
            tuple: (sample字典, 图像文件名)
                - sample字典包含 'image' 和 'label' 键
                - 图像文件名用于标识样本
        """
        # 获取图像文件名
        image_filename = self.images_list[idx]
        
        # 读取输入图像（BGR格式）
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # 将图像调整到目标尺寸
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # 读取标签/掩码图像（灰度图，flags=0表示以灰度模式读取）
        # 注意：标签文件名通过将图像文件名的扩展名改为"jpg"来获取
        mask = cv2.imread(os.path.join(self.output_path, image_filename[:-3] + "jpg"), 0)
        # 将标签调整到目标尺寸
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # 将标签二值化：小于等于0的像素设为0，大于0的像素设为1
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # 修正图像和标签的维度（确保有正确的通道数）
        image, mask = correct_dims(image, mask)
        
        # 创建样本字典
        sample = {'image': image, 'label': mask}

        # 应用数据增强变换（如果有的话）
        if self.joint_transform:
            sample = self.joint_transform(sample)

        # 如果需要，将标签转换为one-hot编码格式
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            # 创建one-hot编码：创建形状为 (类别数, H, W) 的零张量
            # 然后使用scatter_将对应位置的标签值设为1
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
            sample['label'] = mask

        return sample, image_filename

