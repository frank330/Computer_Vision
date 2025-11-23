"""
Dice系数损失函数模块

该模块实现了用于图像分割任务的Dice损失函数和相关的辅助函数。
Dice损失函数常用于处理类别不平衡的分割任务，特别适合医学图像分割。

主要函数：
- build_target: 将目标标签转换为one-hot编码格式，用于Dice损失计算
- dice_coeff: 计算单个类别的Dice系数
- multiclass_dice_coeff: 计算多类别的平均Dice系数
- dice_loss: Dice损失函数（1 - Dice系数）
"""

import torch                         # PyTorch核心库
import torch.nn as nn                # PyTorch神经网络模块


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """
    构建用于Dice系数计算的目标标签（one-hot编码格式）
    
    该函数将分割标签转换为one-hot编码格式，以便计算Dice系数。
    对于需要忽略的像素（如padding区域），会进行特殊处理。
    
    Args:
        target (torch.Tensor): 原始分割标签，形状为 (N, H, W)
            N: batch size
            H, W: 图像高度和宽度
            标签值应该在 [0, num_classes-1] 范围内，ignore_index 表示需要忽略的像素
        num_classes (int): 分割类别数（包括背景），默认2
        ignore_index (int): 需要忽略的像素值，默认-100
            如果 >= 0，则会在one-hot编码中将这些位置设为0
        
    Returns:
        torch.Tensor: one-hot编码的目标标签，形状为 (N, C, H, W)
            C: num_classes（类别数）
    """
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        # 将忽略位置临时设为0，以便进行one_hot编码
        dice_target[ignore_mask] = 0
        # 确保所有值都在有效范围内 [0, num_classes-1]
        dice_target = torch.clamp(dice_target, 0, num_classes - 1)
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        # 将忽略位置的所有通道设为0（在dice_loss中会通过ignore_index参数忽略这些位置）
        # 注意：这里保持one_hot编码格式，忽略位置在dice_loss计算时会被过滤
        if ignore_mask.any():
            # 扩展ignore_mask到one_hot编码的维度 [N, H, W] -> [N, H, W, C]
            ignore_mask_expanded = ignore_mask.unsqueeze(-1).expand_as(dice_target)
            dice_target[ignore_mask_expanded] = 0.0
    else:
        # 确保所有值都在有效范围内
        dice_target = torch.clamp(dice_target, 0, num_classes - 1)
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """
    计算Dice系数（单个类别）
    
    Dice系数用于衡量两个集合的相似度，在图像分割中用于衡量预测结果与真实标签的重叠程度。
    Dice系数 = 2 * |A ∩ B| / (|A| + |B|)
    
    该函数计算一个batch中所有图像某个类别的平均Dice系数。
    
    Args:
        x (torch.Tensor): 预测结果（经过softmax的概率图），形状为 (N, H, W)
            N: batch size
            H, W: 图像高度和宽度
            值应该在 [0, 1] 范围内
        target (torch.Tensor): 真实标签（one-hot编码），形状为 (N, H, W)
            值应该在 [0, 1] 范围内
        ignore_index (int): 需要忽略的像素值，默认-100
            如果 >= 0，则会在计算时忽略这些像素
        epsilon (float): 平滑项，防止分母为0，默认1e-6
    
    Returns:
        float: 平均Dice系数，范围 [0, 1]
            1表示完全匹配，0表示完全不匹配
    """
    d = 0.  # 累积的Dice系数总和
    batch_size = x.shape[0]
    
    # 遍历batch中的每个样本
    for i in range(batch_size):
        # 将预测和标签展平为一维向量
        x_i = x[i].reshape(-1)  # 预测值，形状 (H*W,)
        t_i = target[i].reshape(-1)  # 真实标签，形状 (H*W,)
        
        # 如果需要忽略某些像素
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域（有效区域）
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]  # 只保留有效区域的预测值
            t_i = t_i[roi_mask]  # 只保留有效区域的标签值
        
        # 计算交集：预测值和标签值的点积（对应位置相乘后求和）
        inter = torch.dot(x_i, t_i)
        
        # 计算并集：预测值的总和 + 标签值的总和
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        
        # 如果并集为0（两个集合都为空），则Dice系数定义为1（完全匹配）
        if sets_sum == 0:
            sets_sum = 2 * inter
        
        # 计算Dice系数：2 * 交集 / 并集
        # 添加epsilon防止分母为0
        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    # 返回batch的平均Dice系数
    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """
    计算多类别的平均Dice系数
    
    该函数计算所有类别的Dice系数，然后取平均值。
    用于多类别分割任务的评估。
    
    Args:
        x (torch.Tensor): 预测结果（经过softmax的概率图），形状为 (N, C, H, W)
            N: batch size
            C: 类别数
            H, W: 图像高度和宽度
        target (torch.Tensor): 真实标签（one-hot编码），形状为 (N, C, H, W)
        ignore_index (int): 需要忽略的像素值，默认-100
        epsilon (float): 平滑项，防止分母为0，默认1e-6
    
    Returns:
        float: 所有类别的平均Dice系数，范围 [0, 1]
    """
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    """
    Dice损失函数
    
    Dice损失 = 1 - Dice系数
    用于训练分割模型，目标是最小化该损失（即最大化Dice系数）。
    
    Args:
        x (torch.Tensor): 模型输出的logits，形状为 (N, C, H, W)
            N: batch size
            C: 类别数
            H, W: 图像高度和宽度
        target (torch.Tensor): 真实标签（one-hot编码），形状为 (N, C, H, W)
        multiclass (bool): 是否为多类别分割，默认False
            True: 使用多类别Dice系数
            False: 使用单类别Dice系数
        ignore_index (int): 需要忽略的像素值，默认-100
    
    Returns:
        torch.Tensor: Dice损失值，范围 [0, 1]
            0表示完全匹配，1表示完全不匹配
    """
    # 对logits应用softmax，转换为概率分布
    x = nn.functional.softmax(x, dim=1)
    
    # 根据是否为多类别选择相应的Dice系数计算函数
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    
    # 返回Dice损失（1 - Dice系数）
    return 1 - fn(x, target, ignore_index=ignore_index)
