# -*- coding: utf-8 -*-
"""
单轮训练模块
该模块实现了训练和验证一个epoch的完整流程，包括前向传播、损失计算、反向传播、指标计算和日志记录。
"""

import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    """
    打印训练/验证的摘要信息
    
    功能：
        - 格式化并输出当前batch的训练/验证信息
        - 包括损失、Dice系数、IoU、学习率、时间等指标
    
    Args:
        epoch: 当前epoch编号
        i: 当前batch编号
        nb_batch: 总batch数
        loss: 当前batch的损失值
        loss_name: 损失函数名称
        batch_time: 当前batch的处理时间
        average_loss: 平均损失值
        average_time: 平均处理时间
        iou: 当前batch的IoU值
        average_iou: 平均IoU值
        dice: 当前batch的Dice系数
        average_dice: 平均Dice系数
        acc: 当前batch的准确率（未使用）
        average_acc: 平均准确率（未使用）
        mode: 模式（'Train' 或 'Val'）
        lr: 当前学习率
        logger: 日志记录器对象
    """
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    # string += 'IoU:{:.3f} '.format(iou)
    # string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          训练/验证一个Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger):
    """
    训练或验证一个完整的epoch
    
    功能：
        - 遍历数据加载器中的所有batch
        - 进行前向传播、计算损失、反向传播（训练模式）
        - 计算并累积各种评估指标（Dice、IoU等）
        - 记录日志和TensorBoard信息
        - 更新学习率（如果提供了学习率调度器）
    
    Args:
        loader: 数据加载器（DataLoader对象）
        model: 模型对象
        criterion: 损失函数
        optimizer: 优化器（验证模式下不使用，但保留用于兼容性）
        writer: TensorBoard写入器（可选，如果为None则不记录）
        epoch: 当前epoch编号
        lr_scheduler: 学习率调度器（可选，如果为None则不更新学习率）
        logger: 日志记录器对象
    
    Returns:
        tuple: (average_loss, train_dice_avg, train_iou_average)
            - average_loss: 平均损失值
            - train_dice_avg: 平均Dice系数
            - train_iou_average: 平均IoU值
    """
    # 根据模型模式确定日志模式（训练或验证）
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()  # 记录开始时间
    time_sum, loss_sum = 0, 0  # 累积时间和损失
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0  # 累积评估指标

    dices = []  # 存储每个batch的Dice值
    for i, (sampled_batch, names) in enumerate(loader, 1):

        # 获取损失函数名称
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # 将数据移到GPU上
        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()

        # ====================================================
        #            前向传播和损失计算
        # ====================================================

        # 模型前向传播
        preds = model(images)
        # 计算损失
        out_loss = criterion(preds, masks.float())

        # ====================================================
        #            反向传播和参数更新（仅训练模式）
        # ====================================================
        if model.training:
            optimizer.zero_grad()  # 清零梯度
            out_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        # ====================================================
        #            计算评估指标
        # ====================================================
        # 计算IoU（交并比）
        train_iou = iou_on_batch(masks, preds)
        # 计算Dice系数
        train_dice = criterion._show_dice(preds, masks.float())

        # 计算batch处理时间
        batch_time = time.time() - end
        
        # 如果达到可视化频率且是验证模式，保存可视化结果
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, masks, preds, names, vis_path)

        dices.append(train_dice)  # 保存当前batch的Dice值

        # 累积指标（按样本数量加权，处理最后一个batch可能不满batch_size的情况）
        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        dice_sum += len(images) * train_dice

        # 计算平均值（考虑最后一个batch可能不满batch_size）
        if i == len(loader):
            # 最后一个batch：使用实际样本数计算
            total_samples = config.batch_size * (i - 1) + len(images)
            average_loss = loss_sum / total_samples
            average_time = time_sum / total_samples
            train_iou_average = iou_sum / total_samples
            train_dice_avg = dice_sum / total_samples
        else:
            # 非最后一个batch：使用标准batch_size计算
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()  # 更新结束时间
        torch.cuda.empty_cache()  # 清空GPU缓存

        # 按配置的频率打印摘要信息
        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        # 如果启用了TensorBoard，记录指标
        if config.tensorboard and writer is not None:
            step = epoch * len(loader) + i  # 计算全局步数
            # 记录损失
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)
            # 记录IoU
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # 记录Dice系数
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()  # 再次清空GPU缓存

    # ====================================================
    #            更新学习率（如果提供了调度器）
    # ====================================================
    if lr_scheduler is not None:
        lr_scheduler.step()  # 更新学习率
    
    # 返回epoch的平均指标
    return average_loss, train_dice_avg, train_iou_average

