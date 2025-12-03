# -*- coding: utf-8 -*-
"""
工具函数模块
该模块提供了训练和评估过程中使用的各种工具函数，包括：
1. 损失函数（加权BCE、加权Dice、组合损失等）
2. 评估指标计算函数（Dice、IoU、AUC等）
3. 可视化保存函数
4. 学习率调度器（余弦退火重启）
"""

import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F
import math
from functools import wraps
import warnings
import weakref
from torch.optim.optimizer import Optimizer

class WeightedBCE(nn.Module):
    """
    加权二值交叉熵损失函数
    
    功能：
        - 对正样本和负样本分别设置不同的权重
        - 通过权重平衡正负样本的贡献，适用于类别不平衡的数据集
    
    使用场景：
        当数据集中正样本（前景）和负样本（背景）数量不平衡时，
        可以通过调整weights参数来平衡两者的损失贡献
    """

    def __init__(self, weights=[0.4, 0.6]):
        """
        初始化加权BCE损失函数
        
        Args:
            weights: 权重列表 [正样本权重, 负样本权重]
                默认 [0.4, 0.6] 表示负样本权重更高
        """
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        """
        前向传播计算加权BCE损失
        
        Args:
            logit_pixel: 模型预测输出（logits，未经过sigmoid）
            truth_pixel: 真实标签（ground truth）
        
        Returns:
            torch.Tensor: 加权后的BCE损失值
        """
        # 将输入展平为一维向量
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape == truth.shape)
        
        # 计算标准BCE损失（不进行reduction，保留每个像素的损失）
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        
        # 分离正样本和负样本
        pos = (truth > 0.5).float()  # 正样本掩码
        neg = (truth < 0.5).float()  # 负样本掩码
        
        # 计算正负样本的数量（用于归一化）
        pos_weight = pos.sum().item() + 1e-12  # 避免除零
        neg_weight = neg.sum().item() + 1e-12
        
        # 加权损失：正样本和负样本分别加权并归一化
        loss = (self.weights[0] * pos * loss / pos_weight + 
                self.weights[1] * neg * loss / neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    """
    加权Dice损失函数
    
    功能：
        - 计算预测和真实标签之间的Dice相似系数
        - 通过权重对不同区域进行加权
        - Dice损失 = 1 - Dice系数（越小越好）
    
    使用场景：
        用于图像分割任务，特别适合处理类别不平衡的问题
        Dice损失关注重叠区域，对边界敏感
    """
    def __init__(self, weights=[0.5, 0.5]):
        """
        初始化加权Dice损失函数
        
        Args:
            weights: 权重列表 [正样本权重, 负样本权重]
                默认 [0.5, 0.5] 表示正负样本权重相等
        """
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        """
        前向传播计算加权Dice损失
        
        Args:
            logit: 模型预测输出（经过sigmoid，值在[0,1]之间）
            truth: 真实标签（值在[0,1]之间）
            smooth: 平滑系数，防止除零（默认1e-5）
        
        Returns:
            torch.Tensor: Dice损失值（标量）
        """
        batch_size = len(logit)
        # 将输入展平为 [batch_size, H*W]
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(logit.shape == truth.shape)
        
        p = logit.view(batch_size, -1)  # 预测值
        t = truth.view(batch_size, -1)  # 真实值
        
        # 根据真实标签计算权重：正样本区域权重高，负样本区域权重低
        w = truth.detach()  # 分离计算图，不参与梯度计算
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        
        # 应用权重
        p = w * (p)
        t = w * (t)
        
        # 计算Dice系数：Dice = 2*|A∩B| / (|A| + |B|)
        intersection = (p * t).sum(-1)  # 交集
        union = (p * p).sum(-1) + (t * t).sum(-1)  # 并集（近似）
        dice = 1 - (2 * intersection + smooth) / (union + smooth)  # Dice损失

        loss = dice.mean()  # 对batch求平均
        return loss

class WeightedDiceBCE(nn.Module):
    """
    组合损失函数：加权Dice损失 + 加权BCE损失
    
    功能：
        - 结合Dice损失和BCE损失的优点
        - Dice损失关注重叠区域，BCE损失关注逐像素分类
        - 通过权重平衡两种损失的贡献
    
    使用场景：
        图像分割任务中常用的组合损失函数，
        能够同时优化重叠区域和逐像素分类精度
    """
    def __init__(self, dice_weight=1, BCE_weight=1):
        """
        初始化组合损失函数
        
        Args:
            dice_weight: Dice损失的权重（默认1.0）
            BCE_weight: BCE损失的权重（默认1.0）
        """
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])  # 加权BCE损失
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])  # 加权Dice损失
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        """
        计算硬Dice系数（用于评估，不用于训练）
        
        功能：
            - 将预测和标签二值化后计算Dice系数
            - 用于评估模型性能
        
        Args:
            inputs: 模型预测输出（值在[0,1]之间）
            targets: 真实标签（值在[0,1]之间）
        
        Returns:
            float: Dice系数（值越大越好，范围[0,1]）
        """
        # 二值化预测结果：大于等于0.5的设为1，否则设为0
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        # 二值化真实标签：大于0的设为1，否则设为0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        # 计算Dice系数：1 - Dice损失 = Dice系数
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        """
        前向传播计算组合损失
        
        Args:
            inputs: 模型预测输出（logits或经过sigmoid的输出）
            targets: 真实标签
        
        Returns:
            torch.Tensor: 组合损失值 = dice_weight * Dice损失 + BCE_weight * BCE损失
        """
        # 分别计算Dice损失和BCE损失
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        
        # 加权组合两种损失
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss

def auc_on_batch(masks, pred):
    """
    计算一个batch的平均AUC（ROC曲线下面积）
    
    功能：
        - 对batch中的每个样本计算AUC
        - 返回batch的平均AUC值
    
    Args:
        masks: 真实标签（torch.Tensor），形状为 [batch_size, H, W] 或 [batch_size, 1, H, W]
        pred: 模型预测输出（torch.Tensor），形状为 [batch_size, 1, H, W]
    
    Returns:
        float: batch的平均AUC值（范围[0,1]，越大越好）
    """
    aucs = []
    for i in range(pred.shape[0]):  # 遍历batch中的每个样本
        # 将预测结果转换为numpy数组（移除梯度信息）
        prediction = pred[i][0].cpu().detach().numpy()
        # 将真实标签转换为numpy数组
        mask = masks[i].cpu().detach().numpy()
        # 计算AUC（需要展平为一维数组）
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)

def iou_on_batch(masks, pred):
    """
    计算一个batch的平均IoU（交并比）
    
    功能：
        - 对batch中的每个样本计算IoU
        - 返回batch的平均IoU值
        - IoU = |A ∩ B| / |A ∪ B|，衡量预测和真实标签的重叠程度
    
    Args:
        masks: 真实标签（torch.Tensor），形状为 [batch_size, H, W] 或 [batch_size, 1, H, W]
        pred: 模型预测输出（torch.Tensor），形状为 [batch_size, 1, H, W]，值在[0,1]之间
    
    Returns:
        float: batch的平均IoU值（范围[0,1]，越大越好）
    """
    ious = []

    for i in range(pred.shape[0]):  # 遍历batch中的每个样本
        # 将预测结果转换为numpy数组
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # 将真实标签转换为numpy数组
        mask_tmp = masks[i].cpu().detach().numpy()
        
        # 二值化预测结果：大于等于0.5的设为1，否则设为0
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        # 二值化真实标签：大于0的设为1，否则设为0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        
        # 计算IoU（Jaccard系数）
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)

def dice_coef(y_true, y_pred):
    """
    计算Dice系数
    
    功能：
        - 计算两个二值化数组之间的Dice相似系数
        - Dice = 2*|A∩B| / (|A| + |B|)
    
    Args:
        y_true: 真实标签（numpy数组）
        y_pred: 预测结果（numpy数组）
    
    Returns:
        float: Dice系数（范围[0,1]，越大越好）
    """
    smooth = 1e-5  # 平滑系数，防止除零
    y_true_f = y_true.flatten()  # 展平为一维数组
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)  # 计算交集
    # 计算Dice系数
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_on_batch(masks, pred):
    """
    计算一个batch的平均Dice系数
    
    功能：
        - 对batch中的每个样本计算Dice系数
        - 返回batch的平均Dice值
    
    Args:
        masks: 真实标签（torch.Tensor），形状为 [batch_size, H, W] 或 [batch_size, 1, H, W]
        pred: 模型预测输出（torch.Tensor），形状为 [batch_size, 1, H, W]，值在[0,1]之间
    
    Returns:
        float: batch的平均Dice系数（范围[0,1]，越大越好）
    """
    dices = []

    for i in range(pred.shape[0]):  # 遍历batch中的每个样本
        # 将预测结果转换为numpy数组
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # 将真实标签转换为numpy数组
        mask_tmp = masks[i].cpu().detach().numpy()
        
        # 二值化预测结果：大于等于0.5的设为1，否则设为0
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        # 二值化真实标签：大于0的设为1，否则设为0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        
        # 计算Dice系数
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)

def save_on_batch(images1, masks, pred, names, vis_path):
    """
    保存一个batch的可视化结果
    
    功能：
        - 将预测结果和真实标签保存为图像文件
        - 用于训练过程中的可视化检查
    
    Args:
        images1: 原始图像（未使用，保留用于兼容性）
        masks: 真实标签（torch.Tensor），形状为 [batch_size, H, W] 或 [batch_size, 1, H, W]
        pred: 模型预测输出（torch.Tensor），形状为 [batch_size, 1, H, W]
        names: 图像文件名列表
        vis_path: 可视化结果保存路径
    """
    for i in range(pred.shape[0]):  # 遍历batch中的每个样本
        # 将预测结果转换为numpy数组
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # 将真实标签转换为numpy数组
        mask_tmp = masks[i].cpu().detach().numpy()
        
        # 二值化并转换为0-255范围：预测结果
        pred_tmp[pred_tmp >= 0.5] = 255
        pred_tmp[pred_tmp < 0.5] = 0
        # 二值化并转换为0-255范围：真实标签
        mask_tmp[mask_tmp > 0] = 255
        mask_tmp[mask_tmp <= 0] = 0

        # 保存预测结果图像（文件名：原文件名_pred.jpg）
        cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
        # 保存真实标签图像（文件名：原文件名_gt.jpg）
        cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)



class _LRScheduler(object):
    """
    学习率调度器基类
    
    功能：
        - 提供学习率调度的基础功能
        - 确保学习率调度器在优化器更新之后调用
        - 管理学习率的状态和恢复
    
    注意：
        这是内部基类，不应该直接使用，应该使用其子类如CosineAnnealingWarmRestarts
    """

    def __init__(self, optimizer, last_epoch=-1):
        """
        初始化学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            last_epoch: 上一个epoch的索引（用于恢复训练，默认-1表示从头开始）
        """
        # 检查优化器类型
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # 初始化epoch和基础学习率
        if last_epoch == -1:
            # 如果是新训练，将当前学习率设置为初始学习率
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            # 如果是恢复训练，检查是否指定了初始学习率
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        # 保存所有参数组的基础学习率
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # 以下代码确保 lr_scheduler.step() 在 optimizer.step() 之后调用
        # 参考：https://github.com/pytorch/pytorch/issues/20124
        def with_counter(method):
            """
            包装优化器的step方法，添加计数器以跟踪调用次数
            用于检测学习率调度器和优化器的调用顺序
            """
            if getattr(method, '_with_counter', False):
                # 如果已经包装过，直接返回
                return method

            # 使用弱引用避免循环引用
            instance_ref = weakref.ref(method.__self__)
            # 获取未绑定的方法
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # 增加调用计数
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # 标记已包装
            wrapper._with_counter = True
            return wrapper

        # 包装优化器的step方法
        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0  # 初始化优化器调用计数
        self._step_count = 0  # 初始化调度器调用计数

        self.step()  # 执行初始step

    def state_dict(self):
        """
        返回调度器的状态字典
        
        功能：
            - 保存调度器的所有状态信息（不包括优化器）
            - 用于模型检查点的保存和恢复
        
        Returns:
            dict: 包含调度器状态的字典
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """
        加载调度器的状态
        
        Args:
            state_dict (dict): 调度器状态字典，应该是由 state_dict() 方法返回的对象
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """
        返回当前调度器计算的上一个学习率
        
        Returns:
            list: 每个参数组的学习率列表
        """
        return self._last_lr

    def get_lr(self):
        """
        计算当前应该使用的学习率（需要在子类中实现）
        
        Raises:
            NotImplementedError: 必须在子类中实现此方法
        """
        raise NotImplementedError

    def step(self, epoch=None):
        """
        更新学习率（执行一步调度）
        
        功能：
            - 根据当前epoch计算新的学习率
            - 更新优化器中所有参数组的学习率
            - 检测并警告不正确的调用顺序
        
        Args:
            epoch: 当前epoch索引（可选，如果不提供则自动递增）
        
        注意：
            - 应该在 optimizer.step() 之后调用
            - 参考：https://github.com/pytorch/pytorch/issues/20124
        """
        # 检测并警告旧的调用模式
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # 检查是否在optimizer.step()之前调用了lr_scheduler.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1  # 增加调度器调用计数

        # 上下文管理器：在调用get_lr()时设置标志
        class _enable_get_lr_call:
            """
            上下文管理器：在调用get_lr()时设置标志
            用于检测get_lr()是否在step()方法内部被调用
            """
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                # 如果没有指定epoch，自动递增
                self.last_epoch += 1
                values = self.get_lr()  # 调用子类实现的get_lr()方法
            else:
                # 如果指定了epoch，使用指定的值
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    # 如果有闭式解，使用闭式解（更快）
                    values = self._get_closed_form_lr()
                else:
                    # 否则使用迭代方法
                    values = self.get_lr()

        # 更新优化器中所有参数组的学习率
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        # 保存当前学习率
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    余弦退火重启学习率调度器（SGDR: Stochastic Gradient Descent with Warm Restarts）
    
    功能：
        - 使用余弦函数周期性地调整学习率
        - 在每个周期结束时"重启"学习率到初始值
        - 通过周期性重启帮助模型跳出局部最优
    
    数学公式：
        η_t = η_min + (1/2) * (η_max - η_min) * (1 + cos(π * T_cur / T_i))
        
        其中：
        - η_max: 初始学习率
        - η_min: 最小学习率
        - T_cur: 当前周期内的epoch数
        - T_i: 当前周期的总epoch数
    
    论文：
        SGDR: Stochastic Gradient Descent with Warm Restarts
        https://arxiv.org/abs/1608.03983
    """


    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        """
        初始化余弦退火重启学习率调度器
        
        Args:
            optimizer: PyTorch优化器对象
            T_0: 第一个重启周期的epoch数
            T_mult: 重启周期倍增因子（每次重启后，周期长度乘以T_mult，默认1表示周期长度不变）
            eta_min: 最小学习率（默认0）
            last_epoch: 上一个epoch的索引（用于恢复训练，默认-1表示从头开始）
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0  # 第一个周期的长度
        self.T_i = T_0  # 当前周期的长度
        self.T_mult = T_mult  # 周期倍增因子
        self.eta_min = eta_min  # 最小学习率

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch  # 当前周期内的epoch数

    def get_lr(self):
        """
        计算当前应该使用的学习率
        
        功能：
            - 根据余弦函数计算当前epoch的学习率
            - 公式：η = η_min + (η_max - η_min) * (1 + cos(π * T_cur / T_i)) / 2
        
        Returns:
            list: 每个参数组的学习率列表
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        # 根据余弦函数计算学习率
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        更新学习率（执行一步调度）
        
        功能：
            - 更新当前周期内的epoch计数
            - 当达到周期长度时，重置计数并增加周期长度（如果T_mult > 1）
            - 计算并更新学习率
        
        Args:
            epoch: 当前epoch索引（可选，如果不提供则自动递增）
        
        注意：
            - 可以在每个batch后调用（传入 epoch + i / iters）
            - 也可以在每个epoch后调用（不传参数或传入epoch）
            - 应该在 optimizer.step() 之后调用
        
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        此函数可以交错调用。

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            # 自动递增模式：epoch自动加1
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1  # 当前周期内的epoch数加1
            # 如果达到当前周期长度，重置计数并增加周期长度
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i  # 重置为0（新周期开始）
                self.T_i = self.T_i * self.T_mult  # 增加周期长度（如果T_mult > 1）
        else:
            # 指定epoch模式：使用传入的epoch值
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                # 已经完成第一个周期
                if self.T_mult == 1:
                    # 如果周期长度不变，使用模运算计算当前周期内的epoch数
                    self.T_cur = epoch % self.T_0
                else:
                    # 如果周期长度变化，计算当前处于第几个周期
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    # 计算当前周期内的epoch数
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    # 更新当前周期长度
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                # 还在第一个周期内
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)  # 更新上一个epoch

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

