# -*- coding: utf-8 -*-
"""
UCTransNet 模型配置文件
该文件包含了模型训练、验证和测试所需的所有配置参数
"""
import os
import torch
import time
import ml_collections

##########################################################################
# 模型训练基础参数配置
##########################################################################
# 是否保存训练好的模型
save_model = True
# 是否启用 TensorBoard 记录训练过程
tensorboard = True
# 指定使用的 GPU 设备编号（"0" 表示使用第一块 GPU，多个 GPU 可用逗号分隔，如 "0,1"）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 检查 CUDA 是否可用
use_cuda = torch.cuda.is_available()
# 随机种子，用于保证实验的可重复性
seed = 666
# 设置 Python 哈希种子，确保随机性可复现
os.environ['PYTHONHASHSEED'] = str(seed)

##########################################################################
# 训练策略参数
##########################################################################
# 是否使用余弦学习率衰减策略（True: 使用余弦衰减, False: 使用其他学习率策略）
cosineLR = True
# 输入图像的通道数（3 表示 RGB 彩色图像）
n_channels = 3
# 输出标签的通道数（1 表示二分类分割任务）
n_labels = 1
# 训练的总轮数（epochs）
epochs = 60
# 输入图像的尺寸（224x224 像素）
img_size = 224
# 打印训练信息的频率（每 N 个 epoch 打印一次）
print_frequency = 10
# 模型保存频率（每 N 个 epoch 保存一次模型）
save_frequency = 20
# 可视化验证结果的频率（每 N 个 epoch 可视化一次）
vis_frequency = 10
# 早停机制的耐心值（如果验证损失连续 N 个 epoch 没有改善，则停止训练）
early_stopping_patience = 10

##########################################################################
# 模型预训练配置
##########################################################################
# 是否使用预训练模型（False: 从头开始训练, True: 加载预训练权重）
pretrain = False

##########################################################################
# 优化器参数
##########################################################################
# 学习率（初始学习率）
learning_rate = 1e-3
# 批次大小（每次训练使用的样本数量）
batch_size = 4

##########################################################################
# 模型名称配置
##########################################################################
# 使用的模型名称
model_name = 'UCTransNet'

##########################################################################
# 数据集路径配置
##########################################################################
# 训练集数据路径
train_dataset = './data/training/'
# 验证集数据路径
val_dataset = './data/val/'
# 测试集数据路径
test_dataset = './data/test/'

##########################################################################
# 输出路径配置
##########################################################################
# 会话名称（根据当前时间自动生成，格式：Test_session_MM.DD_HHhMM）
# session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
# 模型保存的根目录路径
save_path          = './log/'
# 模型权重文件保存路径
model_path         = save_path + 'models/'
# TensorBoard 日志文件保存路径
tensorboard_folder = save_path + 'tensorboard_logs/'
# 训练日志文件保存路径
logger_path        = save_path + "train.log"
# 验证结果可视化图片保存路径
visualize_path     = save_path + 'visualize_val/'


##########################################################################
# CTrans Transformer 模型配置
##########################################################################
def get_CTranS_config():
    """
    获取 CTrans Transformer 的配置参数
    
    Returns:
        config (ml_collections.ConfigDict): 包含所有 CTrans 配置的字典对象
    """
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    
    # Key-Value 的尺寸大小（KV_size = Q1 + Q2 + Q3 + Q4，即四个查询层的特征维度之和）
    config.KV_size = 960
    # Transformer 多头注意力机制的头数
    config.transformer.num_heads  = 4
    # Transformer 编码器的层数
    config.transformer.num_layers = 4
    # MLP（多层感知机）通道维度扩展比例
    config.expand_ratio           = 4
    # Embedding 层的 Dropout 比率（防止过拟合）
    config.transformer.embeddings_dropout_rate = 0.1
    # 注意力机制的 Dropout 比率
    config.transformer.attention_dropout_rate = 0.1
    # 其他层的 Dropout 比率
    config.transformer.dropout_rate = 0
    # 图像分块的尺寸列表（用于不同层级的特征提取，从大到小：16, 8, 4, 2）
    config.patch_sizes = [16,8,4,2]
    # U-Net 基础通道数（64 是 U-Net 的起始特征通道数）
    config.base_channel = 64
    # 分类/分割的类别数（1 表示二分类分割任务）
    config.n_classes = 1
    return config


