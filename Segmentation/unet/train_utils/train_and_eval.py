"""
训练和评估函数模块

该模块提供了模型训练和评估的核心函数，包括：
- 损失函数计算（交叉熵 + Dice损失）
- 模型训练（一个epoch）
- 模型评估
- 学习率调度器创建

主要函数：
- criterion: 计算损失函数（交叉熵 + Dice损失）
- train_one_epoch: 训练一个epoch
- evaluate: 在验证集上评估模型
- create_lr_scheduler: 创建学习率调度器
"""

import torch                         # PyTorch核心库
from torch import nn                 # PyTorch神经网络模块
import train_utils.distributed_utils as utils  # 分布式训练工具模块
from .dice_coefficient_loss import dice_loss, build_target  # Dice损失函数


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    """
    计算损失函数（交叉熵 + Dice损失）
    
    该函数结合了交叉熵损失和Dice损失，用于训练分割模型。
    交叉熵损失提供稳定的梯度，Dice损失有助于处理类别不平衡问题。
    
    Args:
        inputs (dict): 模型输出字典，通常包含 'out' 键
            'out': 主输出，形状为 (N, C, H, W)
            'aux': 辅助输出（可选），形状为 (N, C, H, W)
        target (torch.Tensor): 真实标签，形状为 (N, H, W)
            标签值应该在 [0, num_classes-1] 范围内，ignore_index 表示需要忽略的像素
        loss_weight (torch.Tensor, optional): 每个类别的损失权重，用于处理类别不平衡
            形状为 (num_classes,)，默认None（不使用权重）
        num_classes (int): 分割类别数（包括背景），默认2
        dice (bool): 是否使用Dice损失，默认True
            True: 使用交叉熵 + Dice损失
            False: 仅使用交叉熵损失
        ignore_index (int): 需要忽略的像素值，默认-100
            在训练中，这些像素不参与损失计算
    
    Returns:
        torch.Tensor: 总损失值
            如果只有主输出，返回主输出的损失
            如果有辅助输出，返回主输出损失 + 0.5 * 辅助输出损失
    """
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    """
    在验证集/测试集上评估模型
    
    该函数在验证集上评估模型性能，计算混淆矩阵和Dice系数等指标。
    
    Args:
        model (nn.Module): 要评估的模型
        data_loader (DataLoader): 验证集数据加载器
        device (torch.device): 计算设备（'cuda' 或 'cpu'）
        num_classes (int): 分割类别数（包括背景）
    
    Returns:
        tuple: (confmat, dice)
            confmat: 混淆矩阵对象，包含准确率和IoU等指标
            dice: Dice系数（float值）
    """
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    """
    训练模型一个epoch
    
    该函数执行一个完整的训练epoch，包括前向传播、损失计算、反向传播和参数更新。
    支持混合精度训练（AMP）以加速训练并减少显存占用。
    
    Args:
        model (nn.Module): 要训练的模型
        optimizer (torch.optim.Optimizer): 优化器
        data_loader (DataLoader): 训练集数据加载器
        device (torch.device): 计算设备（'cuda' 或 'cpu'）
        epoch (int): 当前epoch编号
        num_classes (int): 分割类别数（包括背景）
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        print_freq (int): 打印训练信息的频率（每N个batch打印一次），默认10
        scaler (torch.cuda.amp.GradScaler, optional): 混合精度训练的梯度缩放器
            如果为None，则不使用混合精度训练
    
    Returns:
        tuple: (mean_loss, lr)
            mean_loss: 该epoch的平均损失值
            lr: 当前学习率
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    """
    创建学习率调度器
    
    该函数创建一个LambdaLR学习率调度器，支持学习率预热（warmup）和多项式衰减策略。
    学习率更新策略参考DeepLab v2的论文。
    
    Args:
        optimizer (torch.optim.Optimizer): 优化器
        num_step (int): 每个epoch的步数（batch数）
        epochs (int): 总训练轮数
        warmup (bool): 是否使用学习率预热，默认True
            True: 训练初期逐渐增加学习率
            False: 不使用预热
        warmup_epochs (int): 预热轮数，默认1
        warmup_factor (float): 预热起始学习率倍率，默认1e-3
            预热过程中，学习率从 warmup_factor * lr 逐渐增加到 lr
    
    Returns:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器
    
    学习率策略：
    1. Warmup阶段：学习率从 warmup_factor * lr 线性增加到 lr
    2. 正常训练阶段：学习率从 lr 多项式衰减到 0
       公式：lr = lr * (1 - (step - warmup_steps) / total_steps) ^ 0.9
    """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子
        
        注意：在训练开始之前，PyTorch会提前调用一次lr_scheduler.step()方法
        所以实际的step数会从1开始（而不是0）
        
        Args:
            x (int): 当前步数（step）
        
        Returns:
            float: 学习率倍率因子
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
