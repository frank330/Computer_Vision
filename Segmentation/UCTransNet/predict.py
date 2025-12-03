# -*- coding: utf-8 -*-
"""
单张图像预测模块
该模块用于对单张图像进行分割预测，确保预处理流程与训练时完全一致。
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import functional as F

# 导入UCTransNet模型和配置
from nets.UCTransNet import UCTransNet
import Config as config


def time_synchronized():
    """
    同步CUDA操作并返回当前时间，用于精确测量推理时间
    
    Returns:
        float: 当前时间戳
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def preprocess_image(image_path, target_size):
    """
    预处理图像，完全模拟训练时的预处理流程
    
    训练时的预处理流程（参考 Load_Dataset.py 的 ImageToImage2D.__getitem__）：
    1. 使用 cv2.imread() 读取图像（BGR格式）
    2. 使用 cv2.resize() 调整尺寸
    3. 使用 correct_dims() 修正维度
    4. 使用 ValGenerator 进行预处理（F.to_pil_image + F.to_tensor）
    
    Args:
        image_path: 输入图像路径
        target_size: 目标图像尺寸 (height, width)
    
    Returns:
        tuple: (预处理后的tensor [1, C, H, W], 原始PIL图像, 原始尺寸)
    """
    # 使用OpenCV读取图像（与训练时保持一致，BGR格式）
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    # 保存原始尺寸和原始图像（用于后续创建叠加图）
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    original_bgr = image.copy()
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_img_pil = Image.fromarray(original_rgb)
    
    # 图像预处理：完全模拟 ImageToImage2D.__getitem__ 的流程
    # 1. 调整图像尺寸（与训练时一致）
    image = cv2.resize(image, (target_size, target_size))
    
    # 2. 修正图像的维度（与训练时一致，使用correct_dims的逻辑）
    # 如果图像是2维的（灰度图），则在第3维添加一个通道维度
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    
    # 3. 创建样本字典（模拟训练时的流程）
    # 注意：这里创建一个虚拟的mask，因为预测时不需要mask
    # 但ValGenerator需要sample字典包含'image'和'label'键
    dummy_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    if len(dummy_mask.shape) == 2:
        dummy_mask = np.expand_dims(dummy_mask, axis=2)
    sample = {'image': image, 'label': dummy_mask}
    
    # 4. 使用ValGenerator进行预处理（与训练时完全一致）
    # ValGenerator会：
    # - 使用F.to_pil_image转换（注意：不会自动转换BGR到RGB）
    # - 检查尺寸，如果已经是目标尺寸则不缩放
    # - 使用F.to_tensor转换
    from Load_Dataset import ValGenerator
    val_generator = ValGenerator(output_size=[target_size, target_size])
    sample = val_generator(sample)
    
    # 5. 提取处理后的图像tensor
    image_tensor = sample['image']  # [C, H, W]
    
    # 6. 添加batch维度：[C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_img_pil, original_size


def predict_single_image(image_path, output_dir="./templates", save_results=True, model_path=None):
    """
    对单张图像进行分割预测
    
    该函数确保预处理流程与训练时完全一致，包括：
    1. 使用相同的图像读取方式（OpenCV）
    2. 使用相同的预处理步骤（resize + to_tensor）
    3. 确保模型处于评估模式
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录，用于保存结果图像（默认 "./templates"）
        save_results: 是否保存结果图像（默认 True）
        model_path: 模型文件路径（默认 None，使用默认路径）
    
    Returns:
        dict: 包含预测结果信息的字典
            - original: 原图路径（如果save_results=True）
            - mask: 分割mask路径（如果save_results=True）
            - overlay: 叠加图路径（如果save_results=True）
            - inference_time: 推理时间（秒）
            - prediction: 预测结果数组（numpy array，值在[0, 255]）
    """
    # 检查输入图像是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 设置模型路径
    if model_path is None:
        model_path = "./log/models/best_model.pth.tar"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}。请指定正确的模型路径。")
    
    print(f"使用模型: {model_path}")
    
    # 获取计算设备（优先使用GPU，如果没有则使用CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取UCTransNet配置
    config_vit = config.get_CTranS_config()
    
    # 创建UCTransNet模型
    model = UCTransNet(config_vit, 
                      n_channels=config.n_channels, 
                      n_classes=config.n_labels,
                      img_size=config.img_size)
    
    # 加载训练好的模型权重
    print("正在加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 加载模型状态字典（兼容不同的保存格式）
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()  # 设置为评估模式（关闭Dropout和BatchNorm的训练模式）
    print("模型加载完成！")
    
    # 预处理图像（与训练时完全一致的流程）
    print("正在预处理图像...")
    img_tensor, original_img_pil, original_size = preprocess_image(image_path, config.img_size)
    print(f"原始图像尺寸: {original_size}")
    print(f"预处理后尺寸: {img_tensor.shape}")
    
    # 确保输出目录存在
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # 进行预测
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速推理
        # 模型预热（可选，用于更准确的计时和GPU初始化）
        init_img = torch.zeros((1, 3, config.img_size, config.img_size), device=device)
        model(init_img)
        
        # 记录推理开始时间
        t_start = time_synchronized()
        
        # 进行模型推理
        output = model(img_tensor.to(device))
        
        # 记录推理结束时间
        t_end = time_synchronized()
        inference_time = t_end - t_start
        print(f"推理时间: {inference_time:.4f}秒")
        
        # UCTransNet的输出已经通过Sigmoid激活，值在[0, 1]之间
        # 使用0.5作为阈值进行二值化
        prediction = (output > 0.5).float()
        
        # 移除batch和channel维度：[1, 1, H, W] -> [H, W]
        prediction = prediction.squeeze(0).squeeze(0)
        
        # 将tensor转换为numpy数组
        prediction = prediction.cpu().numpy()
        
        # 将预测结果调整回原始图像尺寸（使用最近邻插值保持二值化特性）
        prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_NEAREST)
        
        # 将预测结果转换为0-255范围的uint8类型
        prediction = (prediction * 255).astype(np.uint8)
    
    # 如果保存结果
    if save_results:
        # 将numpy数组转换为PIL Image
        mask = Image.fromarray(prediction, mode='L')  # 'L'表示灰度图
        
        # 保存原图（使用原始尺寸的PIL图像，确保RGB格式）
        original_path = os.path.join(output_dir, "original.jpg")
        original_img_pil.save(original_path)
        
        # 保存分割mask（已经是原始尺寸）
        mask_path = os.path.join(output_dir, "mask.jpg")
        mask.save(mask_path)
        
        # 创建叠加图：原图 + 彩色mask叠加
        # 使用原始尺寸的图像和mask，确保尺寸一致
        original_array = np.array(original_img_pil)  # RGB格式的numpy数组，原始尺寸 [H, W, 3]
        mask_array = np.array(mask)  # 灰度图，原始尺寸 [H, W]
        
        # 确保mask_array和original_array的尺寸匹配
        if mask_array.shape[:2] != original_array.shape[:2]:
            # 如果尺寸不匹配，调整mask到原始图像尺寸
            mask_array = cv2.resize(mask_array, (original_array.shape[1], original_array.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # 创建红色叠加（分割区域用红色高亮显示）
        overlay = original_array.copy().astype(np.float32)  # 转换为float以便计算
        
        # 在mask值大于128（白色区域）的位置，叠加红色（透明度0.4）
        mask_bool = mask_array > 128  # 检测mask中的白色区域，形状为 [H, W]
        
        # 对每个通道应用叠加
        for c in range(3):
            if c == 0:  # 红色通道
                overlay[:, :, c][mask_bool] = overlay[:, :, c][mask_bool] * 0.6 + 255 * 0.4
            else:  # 绿色和蓝色通道
                overlay[:, :, c][mask_bool] = overlay[:, :, c][mask_bool] * 0.6
        
        # 转换回uint8类型
        overlay = overlay.astype(np.uint8)
        
        # 保存叠加图
        overlay_img = Image.fromarray(overlay)
        overlay_path = os.path.join(output_dir, "overlay.jpg")
        overlay_img.save(overlay_path)
        
        print(f"\n预测结果已保存:")
        print(f"  原图: {original_path}")
        print(f"  分割mask: {mask_path}")
        print(f"  叠加图: {overlay_path}")
        
        return {
            "original": original_path,
            "mask": mask_path,
            "overlay": overlay_path,
            "inference_time": inference_time,
            "prediction": prediction  # 返回预测结果数组
        }
    else:
        return {
            "inference_time": inference_time,
            "prediction": prediction  # 返回预测结果数组
        }


if __name__ == '__main__':
    """
    主函数：用于测试单张图像预测功能
    
    使用方法：
        python predict.py
    """
    # 示例：预测单张图像
    image_path = r'D:\code\Project\Computer_Vision\Segmentation\UCTransNet-\data\training\images\10.jpg'
    model_path = "./log/models/best_model.pth.tar"

    result = predict_single_image(image_path, model_path=model_path)
    print(f"\n预测完成！推理时间: {result['inference_time']:.4f}秒")
