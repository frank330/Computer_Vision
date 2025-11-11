"""
表情识别模型训练脚本
基于VGG16网络进行表情识别模型的训练和评估
支持7种表情分类：anger, disgust, fear, happy, neutral, sad, surprise
"""

import os
import sys
import torch.optim as optim
from torchvision import transforms, datasets
from utils import read_split_data, train_one_epoch, evaluate
from model import vgg
from my_dataset import MyDataSet
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import random


def result_test(real, pred):
    """
    评估模型性能并生成混淆矩阵
    
    参数:
        real: 真实标签列表
        pred: 预测标签列表
    
    功能:
        - 计算准确率、精确率、召回率和F1分数
        - 生成混淆矩阵并保存为图片
    """
    # 计算混淆矩阵
    cv_conf = confusion_matrix(real, pred)
    
    # 计算各种评估指标
    acc = accuracy_score(real, pred)  # 准确率
    precision = precision_score(real, pred, average='weighted')  # 精确率（加权平均）
    recall = recall_score(real, pred, average='weighted')  # 召回率（加权平均）
    f1 = f1_score(real, pred, average='weighted')  # F1分数（加权平均）
    
    # 打印评估结果
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1,))
    
    # 定义7种表情类别标签
    labels11 = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # 创建混淆矩阵显示对象并绘图
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='')
    
    # 保存混淆矩阵图片
    plt.savefig("results/reConfusionMatrix.tif", dpi=400)


def plot_acc(train_acc):
    """
    绘制训练准确率曲线
    
    参数:
        train_acc: 每个epoch的验证准确率列表
    
    功能:
        - 绘制准确率随epoch变化的折线图
        - 保存图片到 results/acc.png
    """
    sns.set(style='darkgrid')  # 设置绘图风格
    plt.figure(figsize=(10, 7))  # 设置图片大小
    x = list(range(len(train_acc)))  # x轴：epoch编号
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')  # 绘制准确率曲线
    
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Acc')  # y轴标签
    plt.legend(loc='best')  # 显示图例
    plt.savefig('results/acc.png', dpi=400)  # 保存图片


def plot_loss(train_loss):
    """
    绘制训练损失曲线
    
    参数:
        train_loss: 每个epoch的训练损失列表
    
    功能:
        - 绘制损失值随epoch变化的折线图
        - 保存图片到 results/loss.png
    """
    sns.set(style='darkgrid')  # 设置绘图风格
    plt.figure(figsize=(10, 7))  # 设置图片大小
    x = list(range(len(train_loss)))  # x轴：epoch编号
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')  # 绘制损失曲线
    
    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    plt.legend(loc='best')  # 显示图例
    plt.savefig('results/loss.png', dpi=400)  # 保存图片




# ==================== 设备配置 ====================
# 检测并设置计算设备（优先使用GPU，否则使用CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # 强制使用CPU（调试时使用）
# print("using {} device.".format(device))

# ==================== 数据路径配置 ====================
# 数据集路径（需要根据实际路径修改）
data_path = r'D:\code\2025\CV\ExpressionRecognition\K60590_ExpressionRecognition\datasets\data\\'

# 读取并划分数据集（训练集和验证集）
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

# ==================== 数据预处理配置 ====================
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(48),  # 随机裁剪并调整大小为48x48（数据增强）
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),  # 调整大小为256x256
        transforms.CenterCrop(48),  # 中心裁剪为48x48
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
    ])
}

# ==================== 数据集实例化 ====================
# 创建训练数据集对象
train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=data_transform["train"])

# 创建验证数据集对象
val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])

# ==================== 数据加载器配置 ====================
batch_size = 16  # 批次大小

# 计算数据加载的工作进程数（取CPU核心数、批次大小和8的最小值）
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
# nw = 0  # 设置为0表示不使用多进程（调试时使用）
# print('Using {} dataloader workers every process'.format(nw))

# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 训练时打乱数据
    pin_memory=True,  # 加速GPU传输
    num_workers=nw,  # 数据加载的工作进程数
    collate_fn=train_dataset.collate_fn  # 自定义的批次整理函数
)

# 创建验证数据加载器
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # 验证时不打乱数据
    pin_memory=True,  # 加速GPU传输
    num_workers=nw,  # 数据加载的工作进程数
    collate_fn=val_dataset.collate_fn  # 自定义的批次整理函数
)

# 获取验证集样本数量
val_num = len(val_dataset)

def train():
    """
    训练VGG16表情识别模型
    
    功能:
        - 加载预训练的VGG16模型
        - 修改分类器以适应7类表情分类
        - 使用迁移学习进行微调
        - 保存最佳模型权重
        - 绘制训练曲线
    """
    # ==================== 模型初始化 ====================
    # 创建VGG16模型实例
    net = vgg(model_name="vgg16")

    # 修改分类器最后一层，将输出类别数改为7（7种表情）
    net.classifier[6] = nn.Linear(in_features=4096, out_features=7)

    # ==================== 加载预训练权重 ====================
    # 预训练权重文件路径
    model_weight_path = "vgg16-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # 加载预训练权重字典
    pretrained_dict = torch.load(model_weight_path, map_location='cpu')
    model_dict = net.state_dict()  # 获取当前模型的状态字典

    # 过滤掉不匹配的键（只保留形状匹配的权重）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict and v.size() == model_dict[k].size()}

    # 更新模型字典，用预训练权重覆盖对应层
    model_dict.update(pretrained_dict)

    # 加载更新后的状态字典
    net.load_state_dict(model_dict)

    # 将模型移动到指定设备（GPU或CPU）
    net.to(device)

    # ==================== 损失函数和优化器 ====================
    # 定义损失函数（交叉熵损失，适用于多分类问题）
    loss_function = nn.CrossEntropyLoss()

    # 构造优化器（Adam优化器，只优化需要梯度的参数）
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)  # 学习率设置为0.0001

    # ==================== 训练配置 ====================
    epochs = 50  # 训练轮数
    best_acc = 0.0  # 记录最佳验证准确率
    save_path = './ModelFile/vgg.pth'  # 模型保存路径
    train_steps = len(train_loader)  # 每个epoch的训练步数
    acc11 = []  # 记录每个epoch的验证准确率
    loss11 = []  # 记录每个epoch的训练损失

    # ==================== 训练循环 ====================
    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        net.train()  # 设置为训练模式
        running_loss = 0.0  # 累计损失
        
        # 创建训练进度条
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, data in enumerate(train_bar):
            images, labels = data  # 获取图像和标签
            
            optimizer.zero_grad()  # 清零梯度
            logits = net(images.to(device))  # 前向传播，获取预测结果
            loss = loss_function(logits, labels.to(device))  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            # 累计损失
            running_loss += loss.item()

            # 更新进度条描述
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # ---------- 验证阶段 ----------
        net.eval()  # 设置为评估模式
        acc = 0.0  # 累计正确预测数量
        
        with torch.no_grad():  # 验证时不需要计算梯度
            # 创建验证进度条
            val_bar = tqdm(val_loader, file=sys.stdout)
            
            for val_data in val_bar:
                val_images, val_labels = val_data  # 获取验证图像和标签
                outputs = net(val_images.to(device))  # 前向传播
                
                # 获取预测类别（最大概率对应的类别）
                predict_y = torch.max(outputs, dim=1)[1]
                
                # 统计正确预测的数量
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                # 更新进度条描述
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        # 计算验证准确率
        val_accurate = acc / val_num
        
        # 记录当前epoch的准确率和损失
        acc11.append(val_accurate)
        loss11.append(running_loss / train_steps)
        
        # 打印当前epoch的训练结果
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 如果当前验证准确率更高，保存模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f'模型已保存，当前最佳准确率: {best_acc:.4f}')
    
    # ==================== 绘制训练曲线 ====================
    # 绘制准确率曲线
    plot_acc(acc11)
    # 绘制损失曲线
    plot_loss(loss11)
    
    print('Finished Training')  # 训练完成
    print(f'最佳验证准确率: {best_acc:.4f}')

def evals():
    """
    评估训练好的模型
    
    功能:
        - 加载训练好的模型权重
        - 在验证集上进行预测
        - 计算并显示评估指标
        - 生成混淆矩阵
    """
    # ==================== 模型初始化 ====================
    # 创建VGG16模型，指定7个输出类别，不初始化权重（使用预训练权重）
    net = vgg(model_name="vgg16", num_classes=7, init_weights=False).to(device)

    # ==================== 加载模型权重 ====================
    # 模型权重文件路径
    weights_path = "./ModelFile/vgg.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    
    # 加载训练好的模型权重
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # ==================== 模型评估 ====================
    net.eval()  # 设置为评估模式
    labels = []  # 存储真实标签
    predicts = []  # 存储预测标签
    
    with torch.no_grad():  # 评估时不需要计算梯度
        # 创建验证进度条
        val_bar = tqdm(val_loader, file=sys.stdout)
        
        for val_data in val_bar:
            val_images, val_labels = val_data  # 获取验证图像和标签
            outputs = net(val_images.to(device))  # 前向传播，获取预测结果
            
            # 获取预测类别（最大概率对应的类别索引）
            predict_y = torch.max(outputs, dim=1)[1]
            
            # 将真实标签转换为numpy列表
            val_labels = val_labels.to(device).data.cpu().numpy().tolist()
            for i in val_labels:
                labels.append(i)  # 添加到真实标签列表
            
            # 将预测标签转换为numpy数组
            predict_y = predict_y.data.cpu().numpy()
            for i in predict_y:
                predicts.append(i)  # 添加到预测标签列表

    # ==================== 生成评估报告 ====================
    # 调用评估函数，计算指标并生成混淆矩阵
    result_test(labels, predicts)










# ==================== 主函数 ====================
if __name__ == '__main__':
    # 选择执行训练或评估
    # train()  # 训练模型
    evals()  # 评估模型

