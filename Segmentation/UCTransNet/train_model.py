# -*- coding: utf-8 -*-
import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from nets.UCTransNet import UCTransNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE


def logger_config(log_path):
    """
    配置日志记录器
    
    功能：
        - 创建日志记录器，同时输出到文件和控制台
        - 设置日志级别为INFO
        - 文件输出使用UTF-8编码，支持中文
    
    Args:
        log_path (str): 日志文件保存路径
    
    Returns:
        logging.Logger: 配置好的日志记录器对象
    """
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    
    # 文件处理器：将日志写入文件
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    # 控制台处理器：将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    # 添加处理器到日志记录器
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    """
    保存模型检查点
    
    功能：
        - 保存当前训练状态的模型
        - 如果是最佳模型，保存为 best_model.pth.tar
        - 如果是普通检查点，保存为 model-{epoch}.pth.tar
    
    Args:
        state (dict): 包含模型状态的字典，包括：
            - epoch: 当前epoch编号
            - best_model: 是否为最佳模型（bool）
            - state_dict: 模型权重字典
            - val_loss: 验证损失
            - optimizer: 优化器状态字典
        save_path (str): 模型保存路径
    
    注意：
        - 如果保存目录不存在，会自动创建
        - 最佳模型会覆盖之前的最佳模型
    """
    logger.info('\t Saving to {}'.format(save_path))
    
    # 如果保存目录不存在，创建目录
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch编号
    best_model = state['best_model']  # 是否为最佳模型（布尔值）

    # 根据是否为最佳模型，选择不同的文件名
    if best_model:
        # 最佳模型保存为固定文件名
        filename = save_path + '/' + 'best_model.pth.tar'
    else:
        # 普通检查点保存为带epoch编号的文件名
        filename = save_path + '/' + 'model-{:02d}.pth.tar'.format(epoch)
    
    # 保存模型状态
    torch.save(state, filename)


def worker_init_fn(worker_id):
    """
    数据加载器工作进程初始化函数
    
    功能：
        - 为每个数据加载器工作进程设置随机种子
        - 确保不同进程使用不同的随机种子，避免数据重复
        - 保证实验的可复现性
    
    Args:
        worker_id (int): 工作进程ID（从0开始）
    
    注意：
        - 每个worker的随机种子 = 基础种子 + worker_id
        - 这样可以确保不同worker使用不同的随机序列，但整体可复现
    """
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          主训练循环：加载数据、初始化模型、执行训练
#=================================================================================
##################################################################################


def main_loop(batch_size=config.batch_size, tensorboard=True):
    """
    主训练循环函数
    
    功能：
        - 加载训练集和验证集数据
        - 初始化模型、优化器、损失函数
        - 执行多个epoch的训练和验证
        - 保存最佳模型
        - 记录TensorBoard日志
        - 实现早停机制
    
    Args:
        batch_size (int): 批次大小，默认使用config中的值
        tensorboard (bool): 是否启用TensorBoard记录，默认True
    
    Returns:
        torch.nn.Module: 训练完成的模型对象
    
    训练流程：
        1. 数据加载：创建训练集和验证集的数据加载器
        2. 模型初始化：创建UCTransNet模型并移动到GPU
        3. 优化器设置：Adam优化器 + 可选的学习率调度器
        4. 训练循环：
           - 每个epoch执行一次训练和验证
           - 记录指标到TensorBoard
           - 保存最佳模型（基于验证集Dice系数）
           - 检查早停条件
    """
    # =============================================================
    #       1. 加载训练和验证数据
    # =============================================================
    # 训练集数据增强：随机旋转、翻转、缩放等
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    # 验证集数据生成器：仅缩放，无随机增强
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    
    # 创建训练集和验证集数据集
    train_dataset = ImageToImage2D(config.train_dataset, train_tf, image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 训练集打乱顺序
        worker_init_fn=worker_init_fn,  # 工作进程初始化函数
        num_workers=0,  # 数据加载进程数（0表示主进程加载）
        pin_memory=True  # 将数据固定到内存，加速GPU传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 验证集也打乱（可选，通常设为False）
        worker_init_fn=worker_init_fn,
        num_workers=0,
        pin_memory=True
    )

    # =============================================================
    #       2. 初始化模型
    # =============================================================
    lr = config.learning_rate  # 获取学习率
    
    # 获取Transformer配置
    config_vit = config.get_CTranS_config()
    logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
    
    # 创建UCTransNet模型
    model = UCTransNet(
        config_vit,
        n_channels=config.n_channels,  # 输入通道数（RGB=3）
        n_classes=config.n_labels      # 输出类别数（二分类=1）
    )
    
    # 将模型移动到GPU
    model = model.cuda()
    

    # =============================================================
    #       3. 初始化损失函数、优化器和学习率调度器
    # =============================================================
    # 组合损失函数：Dice损失 + 二值交叉熵损失（权重各0.5）
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    
    # Adam优化器：只优化需要梯度的参数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    # 学习率调度器：余弦退火重启（SGDR）
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # 第一个周期长度（epoch数）
            T_mult=1,    # 周期倍增因子（1表示周期长度不变）
            eta_min=1e-4 # 最小学习率
        )
    else:
        lr_scheduler = None  # 不使用学习率调度器
    
    # =============================================================
    #       4. 初始化TensorBoard
    # =============================================================
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: {}'.format(log_dir))
        # 如果日志目录不存在，创建目录
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        # 创建TensorBoard写入器
        writer = SummaryWriter(log_dir)
    else:
        writer = None  # 不使用TensorBoard

    # =============================================================
    #       5. 开始训练循环
    # =============================================================
    max_dice = 0.0  # 记录最佳验证集Dice系数
    best_epoch = 1  # 记录最佳模型所在的epoch
    
    # 循环训练多个epoch
    for epoch in range(config.epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs))
        # logger.info(config.session_name)
        
        # =============================================================
        #       5.1 训练阶段
        # =============================================================
        model.train(True)  # 设置为训练模式（启用Dropout和BatchNorm的训练模式）
        logger.info('Training with batch size : {}'.format(batch_size))
        # 训练一个epoch，返回平均损失、Dice系数和IoU
        train_loss, train_dice, train_iou = train_one_epoch(
            train_loader, model, criterion, optimizer, writer, epoch, None, logger
        )
        
        # =============================================================
        #       5.2 验证阶段
        # =============================================================
        logger.info('Validation')
        with torch.no_grad():  # 验证时禁用梯度计算，节省内存和加速
            model.eval()  # 设置为评估模式（关闭Dropout和BatchNorm的训练模式）
            # 验证一个epoch，返回平均损失、Dice系数和IoU
            # 注意：验证时传入lr_scheduler，用于更新学习率
            val_loss, val_dice, val_iou = train_one_epoch(
                val_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger
            )
        
        # =============================================================
        #       5.3 记录epoch级别的平均指标到TensorBoard
        # =============================================================
        if tensorboard and writer is not None:
            # 记录训练集的平均指标
            writer.add_scalar('Epoch/Train_Loss', train_loss.item() if hasattr(train_loss, 'item') else train_loss, epoch + 1)
            writer.add_scalar('Epoch/Train_Dice', train_dice, epoch + 1)
            writer.add_scalar('Epoch/Train_IoU', train_iou, epoch + 1)
            
            # 记录验证集的平均指标
            writer.add_scalar('Epoch/Val_Loss', val_loss.item() if hasattr(val_loss, 'item') else val_loss, epoch + 1)
            writer.add_scalar('Epoch/Val_Dice', val_dice, epoch + 1)
            writer.add_scalar('Epoch/Val_IoU', val_iou, epoch + 1)
            
            # 记录当前学习率
            current_lr = min(g["lr"] for g in optimizer.param_groups)
            writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch + 1)

        # =============================================================
        #       5.4 保存最佳模型
        # =============================================================
        if val_dice > max_dice:
            # 如果当前验证集Dice系数超过历史最佳值
            if epoch + 1 > 5:  # 至少训练5个epoch后才保存（避免早期不稳定）
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice  # 更新最佳Dice系数
                best_epoch = epoch + 1  # 更新最佳epoch
                # 保存模型检查点
                save_checkpoint({
                    'epoch': epoch,
                    'best_model': True,  # 标记为最佳模型
                    'state_dict': model.state_dict(),  # 模型权重
                    'val_loss': val_loss,  # 验证损失
                    'optimizer': optimizer.state_dict()  # 优化器状态（用于恢复训练）
                }, config.model_path)
        else:
            # 如果Dice系数没有提升，记录日志
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        
        # =============================================================
        #       5.5 早停机制检查
        # =============================================================
        # 计算自最佳模型以来的epoch数
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))
        
        # 如果连续多个epoch没有提升，触发早停
        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break  # 提前结束训练

    return model  # 返回训练完成的模型


if __name__ == '__main__':
    """
    主程序入口
    
    功能：
        - 设置随机种子，确保实验可复现
        - 配置CUDA优化选项
        - 创建日志目录
        - 初始化日志记录器
        - 启动训练流程
    """
    # =============================================================
    #       1. 配置CUDA优化选项
    # =============================================================
    deterministic = False  # 是否使用确定性算法（False表示允许非确定性优化）
    
    if not deterministic:
        # 非确定性模式：允许CUDA优化，提高训练速度
        cudnn.benchmark = True  # 启用benchmark模式，自动选择最优的卷积算法
        cudnn.deterministic = False  # 允许非确定性算法
    else:
        # 确定性模式：保证结果可复现，但速度较慢
        cudnn.benchmark = False  # 禁用benchmark模式
        cudnn.deterministic = True  # 使用确定性算法
    
    # =============================================================
    #       2. 设置随机种子，确保实验可复现
    # =============================================================
    # 设置Python随机数生成器种子
    random.seed(config.seed)
    # 设置NumPy随机数生成器种子
    np.random.seed(config.seed)
    # 设置PyTorch CPU随机数生成器种子
    torch.manual_seed(config.seed)
    # 设置PyTorch GPU随机数生成器种子（单GPU）
    torch.cuda.manual_seed(config.seed)
    # 设置PyTorch GPU随机数生成器种子（多GPU）
    torch.cuda.manual_seed_all(config.seed)
    
    # =============================================================
    #       3. 创建保存目录
    # =============================================================
    # 如果保存路径不存在，创建目录
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    # =============================================================
    #       4. 初始化日志记录器
    # =============================================================
    logger = logger_config(log_path=config.logger_path)
    
    # =============================================================
    #       5. 启动训练流程
    # =============================================================
    # 调用主训练循环，启用TensorBoard记录
    model = main_loop(tensorboard=True)

