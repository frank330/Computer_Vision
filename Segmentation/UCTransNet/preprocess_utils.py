# -*- coding: utf-8 -*-
"""
图像预处理工具模块
该模块提供与训练时完全一致的图像预处理函数，确保训练和预测时使用相同的预处理流程。
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
import Config as config


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


def preprocess_image_for_inference(image_path, target_size=None):
    """
    预处理图像用于推理，确保与训练时的预处理流程完全一致
    
    训练时的预处理流程（参考 Load_Dataset.py）：
    1. 使用 cv2.imread() 读取图像（BGR格式）
    2. 使用 cv2.resize() 调整尺寸
    3. 使用 correct_dims() 修正维度
    4. 转换为PIL Image
    5. 使用 F.to_tensor() 转换为tensor并归一化到[0,1]
    
    Args:
        image_path: 输入图像路径
        target_size: 目标图像尺寸，如果为None则使用config.img_size
    
    Returns:
        tuple: (预处理后的tensor [1, C, H, W], 原始PIL图像, 原始尺寸(width, height))
    """
    if target_size is None:
        target_size = config.img_size
    
    # 使用OpenCV读取图像（与训练时保持一致，BGR格式）
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    # 保存原始尺寸
    original_size = (image_bgr.shape[1], image_bgr.shape[0])  # (width, height)
    
    # 调整图像尺寸（与训练时保持一致）
    image_resized = cv2.resize(image_bgr, (target_size, target_size))
    
    # 将BGR转换为RGB（OpenCV读取的是BGR，需要转换为RGB）
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # 修正图像维度（与训练时保持一致）
    image_rgb = correct_dims(image_rgb)
    
    # 转换为PIL Image以便使用F.to_tensor（与训练时保持一致）
    image_pil = Image.fromarray(image_rgb)
    
    # 使用F.to_tensor转换为tensor并归一化到[0,1]（与训练时完全一致）
    image_tensor = F.to_tensor(image_pil)  # [C, H, W], 值在[0, 1]
    
    # 添加batch维度：[C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    # 同时保存原始PIL图像用于后续保存（RGB格式）
    original_img_pil = Image.fromarray(image_rgb)
    
    return image_tensor, original_img_pil, original_size


def preprocess_image_simple(image_path, target_size=None):
    """
    简化版预处理函数（如果不需要完全模拟训练流程）
    
    使用PIL读取图像，然后转换为tensor。
    注意：此函数与训练时的预处理略有不同，建议使用 preprocess_image_for_inference。
    
    Args:
        image_path: 输入图像路径
        target_size: 目标图像尺寸，如果为None则使用config.img_size
    
    Returns:
        tuple: (预处理后的tensor [1, C, H, W], 原始PIL图像, 原始尺寸(width, height))
    """
    if target_size is None:
        target_size = config.img_size
    
    # 使用PIL读取图像（RGB格式）
    image_pil = Image.open(image_path).convert('RGB')
    original_size = image_pil.size  # (width, height)
    
    # 调整图像尺寸
    image_pil = image_pil.resize((target_size, target_size), Image.BILINEAR)
    
    # 使用F.to_tensor转换为tensor并归一化到[0,1]
    image_tensor = F.to_tensor(image_pil)  # [C, H, W], 值在[0, 1]
    
    # 添加batch维度：[C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_pil, original_size

