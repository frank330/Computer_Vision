# -*- coding: utf-8 -*-
"""
模型测试模块
该模块用于加载训练好的模型并在测试集上进行评估，计算平均Dice系数。

注意：此文件不是pytest测试文件，而是普通的测试脚本。
如果使用pytest运行，请确保pytest配置忽略此文件，或直接使用python运行。

重要：该模块确保预处理流程与训练时完全一致，包括：
1. 使用OpenCV读取图像（BGR格式，与训练时一致）
2. 使用相同的预处理步骤（resize + to_tensor）
3. 确保模型处于评估模式
4. 不使用重复的sigmoid（模型输出已经过sigmoid）
"""
import os
import torch
import numpy as np
import Config as config
from PIL import Image
import cv2
import glob
from torchvision.transforms import functional as F
from nets.UCTransNet import UCTransNet


def compute_dice(pred, target):
    """
    计算Dice系数

    功能：
        - 计算预测结果和真实标签之间的Dice相似系数
        - Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: 预测结果（numpy数组，二值化）
        target: 真实标签（numpy数组，二值化）

    Returns:
        float: Dice系数（范围[0,1]，越大越好）
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()  # 计算交集
    # 如果预测和标签都为空，返回1.0（完全匹配）
    if pred.sum() + target.sum() == 0:
        return 1.0
    return 2. * intersection / (pred.sum() + target.sum())


def evaluate_average_dice(test_dataset_dir, model, device):
    """
    评估测试集的平均Dice系数

    功能：
        - 遍历测试集中的所有图像
        - 对每张图像进行预测
        - 计算每张图像的Dice系数
        - 返回平均Dice系数

    Args:
        test_dataset_dir: 测试数据集目录路径
        model: 训练好的模型
        device: 计算设备（CPU或GPU）
    """
    # 获取图像和标签目录路径
    images_dir = os.path.join(test_dataset_dir, "images")
    masks_dir = os.path.join(test_dataset_dir, "labels")

    # 检查目录是否存在
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"标签目录不存在: {masks_dir}")

    # 获取所有图像文件路径（支持jpg和png格式）
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                         glob.glob(os.path.join(images_dir, "*.png")))

    if len(image_paths) == 0:
        print(f"警告：在 {images_dir} 中未找到任何图像文件")
        return

    print(f"找到 {len(image_paths)} 张测试图像")
    dice_scores = []

    # 遍历每张图像进行预测和评估
    for img_path in image_paths:
        # 获取图像文件名（不含扩展名）
        name = os.path.splitext(os.path.basename(img_path))[0]

        # 尝试查找对应的标签文件（先尝试png，再尝试jpg）
        mask_path = os.path.join(masks_dir, f"{name}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, f"{name}.jpg")
            if not os.path.exists(mask_path):
                print(f"警告：标签未找到，跳过 {name}")
                continue

        # 读取图像和标签（完全模拟训练时的流程）
        # 使用OpenCV读取图像（BGR格式，与训练时一致）
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue

        # 读取标签（使用OpenCV，与训练时一致）
        mask = cv2.imread(mask_path, 0)  # flags=0表示以灰度模式读取
        if mask is None:
            print(f"警告：无法读取标签 {mask_path}，跳过")
            continue

        # 图像预处理：完全模拟 ImageToImage2D.__getitem__ 的流程
        # 1. 调整图像尺寸（与训练时一致）
        image = cv2.resize(image, (config.img_size, config.img_size))

        # 2. 调整标签尺寸（与训练时一致）
        mask = cv2.resize(mask, (config.img_size, config.img_size))

        # 3. 标签二值化：与训练时一致（小于等于0的像素设为0，大于0的像素设为1）
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # 4. 修正图像和标签的维度（与训练时一致，使用correct_dims）
        # 如果图像是2维的（灰度图），则在第3维添加一个通道维度
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        # 5. 创建样本字典（模拟训练时的流程）
        sample = {'image': image, 'label': mask}

        # 6. 使用ValGenerator进行预处理（与训练时完全一致）
        # ValGenerator会：
        # - 使用F.to_pil_image转换（注意：不会自动转换BGR到RGB）
        # - 检查尺寸，如果已经是目标尺寸则不缩放
        # - 使用F.to_tensor转换
        from Load_Dataset import ValGenerator
        val_generator = ValGenerator(output_size=[config.img_size, config.img_size])
        sample = val_generator(sample)

        # 7. 提取处理后的图像和标签
        img_tensor = sample['image'].unsqueeze(0).to(device)  # [1, C, H, W]
        label_tensor = sample['label']  # [H, W] 或 [1, H, W]

        # 8. 将标签转换为numpy数组用于计算Dice
        if len(label_tensor.shape) == 3:
            gt = label_tensor[0].cpu().numpy().astype(np.uint8)
        else:
            gt = label_tensor.cpu().numpy().astype(np.uint8)

        # 模型推理
        with torch.no_grad():
            output = model(img_tensor)
            # UCTransNet的输出已经通过Sigmoid激活（在模型定义中），值在[0, 1]之间
            # 不需要再次使用sigmoid！
            # 直接使用输出进行二值化
            pred = output.cpu().numpy()[0, 0]  # [H, W]
            pred = (pred > 0.5).astype(np.uint8)

        # 计算Dice系数
        dice = compute_dice(pred, gt)
        dice_scores.append(dice)
        print(f"{name}: dice={dice:.4f}")

    # 输出结果统计
    if len(dice_scores) == 0:
        print("错误：未计算到任何dice分数，请检查数据集路径和文件格式")
    else:
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        print(f"\n{'=' * 50}")
        print(f"测试结果统计")
        print(f"{'=' * 50}")
        print(f"测试图像数量: {len(dice_scores)}")
        print(f"平均Dice系数: {avg_dice:.4f}")
        print(f"Dice系数标准差: {std_dice:.4f}")
        print(f"最高Dice系数: {np.max(dice_scores):.4f}")
        print(f"最低Dice系数: {np.min(dice_scores):.4f}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    """
    主函数：加载模型并在测试集上进行评估

    流程：
        1. 设置计算设备（GPU或CPU）
        2. 创建UCTransNet模型
        3. 加载训练好的模型权重
        4. 在测试集上评估并计算平均Dice系数
    """
    print("=" * 50)
    print("UCTransNet 模型测试")
    print("=" * 50)

    # 配置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取UCTransNet配置
    config_vit = config.get_CTranS_config()

    # 创建模型
    print("正在创建模型...")
    model = UCTransNet(config_vit,
                       n_channels=config.n_channels,
                       n_classes=config.n_labels,
                       img_size=config.img_size)

    # 加载模型权重
    model_path = r"D:\code\Project\Computer_Vision\Segmentation\UCTransNet\log\models\best_model.pth.tar"

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}。请指定正确的模型路径。")

    print(f"使用模型: {model_path}")

    # 加载训练好的模型权重
    print("正在加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # 加载模型权重（兼容不同的保存格式）
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # 设置为评估模式（关闭Dropout和BatchNorm的训练模式）
    print("模型加载完成！\n")

    # 在测试集上评估
    evaluate_average_dice(config.test_dataset, model, device)

