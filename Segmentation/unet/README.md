# U-Net 图像分割项目

基于PyTorch实现的U-Net图像分割框架，支持多种编码器（backbone）变体，适用于医学图像分割、语义分割等任务。

## 📋 项目简介

本项目实现了三种U-Net变体，均采用编码器-解码器架构，通过跳跃连接融合多尺度特征：

- **标准U-Net** (`unet.py`): 完全自定义的编码器-解码器结构，灵活可配置
- **MobileNetV3-U-Net** (`mobilenet_unet.py`): 使用MobileNetV3作为编码器，轻量级，适合移动设备
- **VGG16-U-Net** (`vgg_unet.py`): 使用VGG16-BN作为编码器，特征提取能力强，精度高

### 模型对比

| 特性 | 标准U-Net | MobileNetV3-U-Net | VGG16-U-Net |
|------|-----------|-------------------|-------------|
| **编码器类型** | 自定义构建 | MobileNetV3 Large | VGG16-BN |
| **参数量** | 中等 | 少（轻量级） | 多 |
| **推理速度** | 中等 | 快 | 较慢 |
| **精度** | 中等 | 中等 | 较高 |
| **适用场景** | 通用 | 移动设备/资源受限 | 精度优先 |
| **预训练权重** | 不支持 | 支持（ImageNet） | 支持（ImageNet） |
| **输入通道** | 可配置（默认1） | 固定3（RGB） | 固定3（RGB） |

## 🔗 参考资源

本项目主要参考以下开源仓库：
- [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- [PyTorch Vision](https://github.com/pytorch/vision)

## 🛠️ 环境配置

### 系统要求
- **Python**: 3.6/3.7/3.8/3.9+
- **PyTorch**: 1.10+ (推荐 2.0+)
- **操作系统**: Ubuntu/CentOS/Windows
  - Windows暂不支持多GPU训练
- **硬件**: 推荐使用GPU训练（CUDA支持）

### 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

主要依赖包：
- `torch>=2.0.0` - PyTorch深度学习框架
- `torchvision>=0.15.0` - 计算机视觉工具库
- `numpy` - 数值计算库
- `Pillow` - 图像处理库
- `matplotlib` - 可视化库
- `Flask` - Web服务框架（可选，用于web.py）

## 📁 项目结构

```
unet/
├── src/                          # 模型定义代码
│   ├── __init__.py              # 模型导出接口
│   ├── unet.py                  # 标准U-Net模型
│   ├── mobilenet_unet.py        # MobileNetV3-U-Net模型
│   └── vgg_unet.py              # VGG16-U-Net模型
│
├── train_utils/                 # 训练工具模块
│   ├── __init__.py
│   ├── train_and_eval.py        # 训练和评估函数
│   ├── dice_coefficient_loss.py # Dice损失函数
│   └── distributed_utils.py    # 多GPU训练工具
│
├── data/                        # 数据集目录（需自行准备）
│   ├── training/                # 训练集
│   │   ├── images/              # 训练图像
│   │   └── labels/              # 训练标签（掩码）
│   └── test/                    # 测试集
│       ├── images/              # 测试图像
│       └── labels/              # 测试标签
│
├── save_weights/                # 模型权重保存目录
│
├── my_dataset.py                # 自定义数据集类
├── transforms.py                # 数据增强变换
├── train.py                     # 单GPU训练脚本
├── train_multi_GPU.py           # 多GPU训练脚本（如果存在）
├── predict.py                   # 预测脚本
├── plot_training_results.py     # 训练结果可视化
├── web.py                       # Web服务接口（可选）
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明文档
```

## 🚀 快速开始

### 1. 准备数据集

数据集应按照以下结构组织：
```
data/
├── training/
│   ├── images/    # 训练图像
│   └── labels/    # 训练标签（掩码图像）
└── test/
    ├── images/    # 测试图像
    └── labels/    # 测试标签
```

**DRIVE数据集下载地址**：
- 官网: [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)
- 百度云: [https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw](https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw) 密码: `8no8`

### 2. 选择模型

在 `train.py` 中修改模型导入：

```python
# 标准U-Net（默认）
from src import UNet
model = UNet(in_channels=1, num_classes=2, base_c=64)

# MobileNetV3-U-Net（轻量级）
from src import MobileV3Unet
model = MobileV3Unet(num_classes=2, pretrain_backbone=True)

# VGG16-U-Net（高精度）
from src import VGG16UNet
model = VGG16UNet(num_classes=2, pretrain_backbone=True)
```

### 3. 配置训练参数

#### 方式一：在代码中直接修改（推荐）

打开 `train.py`，修改 `TRAIN_CONFIG` 配置：

```python
TRAIN_CONFIG = {
    "data_path": "./data",          # 数据集根目录路径
    "num_classes": 1,               # 类别数（不包括背景）
    "device": "cuda",               # 训练设备: "cuda" 或 "cpu"
    "batch_size": 4,                # 批次大小
    "epochs": 200,                  # 训练轮数
    "lr": 0.01,                     # 初始学习率
    "momentum": 0.9,                # 动量
    "weight_decay": 1e-4,           # 权重衰减
    "print_freq": 1,                # 打印频率
    "resume": "",                   # 恢复训练的检查点路径
    "save_best": True,              # 是否只保存最佳模型
    "amp": False,                   # 是否使用混合精度训练
}
```

#### 方式二：命令行参数（优先级更高）

命令行参数会覆盖代码中的配置：

```bash
python train.py \
    --data-path ./data \
    --num-classes 1 \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.001 \
    --device cuda
```

### 4. 开始训练

```bash
# 单GPU/CPU训练
python train.py

# 使用命令行参数
python train.py --data-path ./data --batch-size 8 --epochs 100
```

### 5. 模型预测

使用训练好的模型进行预测：

```python
# 在predict.py中修改权重路径和图像路径
python predict.py
```

## 📖 详细使用说明

### 训练参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--data-path` | 数据集根目录路径 | `./` | `./data` |
| `--num-classes` | 类别数（不包括背景） | `1` | `2` |
| `--device` | 训练设备 | `cuda` | `cuda` / `cpu` |
| `-b, --batch-size` | 批次大小 | `4` | `8` |
| `--epochs` | 训练轮数 | `200` | `100` |
| `--lr` | 初始学习率 | `0.01` | `0.001` |
| `--momentum` | 动量（SGD优化器） | `0.9` | `0.9` |
| `--wd, --weight-decay` | 权重衰减 | `1e-4` | `1e-4` |
| `--print-freq` | 打印频率（每N个epoch） | `1` | `5` |
| `--resume` | 恢复训练的检查点路径 | `空` | `./save_weights/best_model.pth` |
| `--start-epoch` | 起始轮数 | `0` | `50` |
| `--save-best` | 是否只保存最佳模型 | `True` | `True` / `False` |
| `--amp` | 是否使用混合精度训练 | `False` | `True` |

### 多GPU训练

如果系统有多块GPU，可以使用多GPU训练加速：

```bash
# 使用8块GPU训练
torchrun --nproc_per_node=8 train_multi_GPU.py

# 指定使用特定GPU（例如使用第1块和第4块GPU）
CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py
```

**注意**：Windows系统暂不支持多GPU训练。

### 模型选择指南

根据你的需求选择合适的模型：

1. **标准U-Net** (`UNet`)
   - ✅ 需要灵活配置输入通道数（如单通道医学图像）
   - ✅ 不需要预训练权重
   - ✅ 参数量和精度平衡

2. **MobileNetV3-U-Net** (`MobileV3Unet`)
   - ✅ 移动设备或边缘设备部署
   - ✅ 对推理速度要求高
   - ✅ 资源受限环境
   - ⚠️ 仅支持RGB三通道输入

3. **VGG16-U-Net** (`VGG16UNet`)
   - ✅ 对精度要求高
   - ✅ 有足够的计算资源
   - ✅ 可以使用ImageNet预训练权重
   - ⚠️ 仅支持RGB三通道输入
   - ⚠️ 参数量大，训练和推理较慢

### 断点续训

如果训练中断，可以从检查点恢复训练：

```bash
python train.py --resume ./save_weights/best_model.pth --start-epoch 50
```

### 混合精度训练

使用混合精度训练可以加速训练并减少显存占用：

```bash
python train.py --amp
```

或在配置中设置：
```python
TRAIN_CONFIG = {
    ...
    "amp": True,
}
```

## ⚠️ 注意事项

1. **数据集路径**：确保 `data_path` 指向包含 `training` 和 `test` 文件夹的根目录
2. **权重路径**：预测时确保 `weights_path` 指向正确的模型权重文件
3. **类别数设置**：`num_classes` 不包括背景类别（例如：二分类任务设置为1）
4. **验证集要求**：验证集或测试集必须包含所有类别的样本
5. **参数优先级**：命令行参数 > 代码中的 `TRAIN_CONFIG` 配置
6. **模型兼容性**：
   - MobileNetV3-U-Net 和 VGG16-U-Net 仅支持3通道RGB输入
   - 标准U-Net支持自定义输入通道数

## 📊 预训练权重

### DRIVE数据集预训练权重

使用标准U-Net在DRIVE数据集上训练得到的权重（仅供测试）：
- 百度云链接: [https://pan.baidu.com/s/1BOqkEpgt1XRqziyc941Hcw](https://pan.baidu.com/s/1BOqkEpgt1XRqziyc941Hcw) 密码: `p50a`

### ImageNet预训练权重

MobileNetV3-U-Net 和 VGG16-U-Net 支持使用ImageNet预训练权重：
- 设置 `pretrain_backbone=True` 时自动下载
- 首次使用需要网络连接下载权重文件

## 📚 学习资源

### U-Net网络原理
- B站视频: [https://www.bilibili.com/video/BV1Vq4y127fB/](https://www.bilibili.com/video/BV1Vq4y127fB/)

### 项目代码分析
- B站视频: [https://b23.tv/PCJJmqN](https://b23.tv/PCJJmqN)

## 🎯 网络结构

本项目U-Net默认使用双线性插值作为上采样方法，网络结构如下：

![U-Net结构图](unet.png)

**核心特点**：
- 编码器-解码器（Encoder-Decoder）架构
- 跳跃连接（Skip Connections）融合多尺度特征
- 双线性插值上采样（可切换为转置卷积）

## 🔧 常见问题

### Q1: 训练时显存不足怎么办？
- 减小 `batch_size`
- 启用混合精度训练 `--amp`
- 使用更轻量的模型（如MobileNetV3-U-Net）

### Q2: 如何切换不同的U-Net模型？
在 `train.py` 中修改模型导入和初始化代码，参考"快速开始"部分的模型选择。

### Q3: 支持哪些数据集格式？
目前支持DRIVE数据集格式，其他数据集需要修改 `my_dataset.py` 中的数据加载逻辑。

### Q4: 如何可视化训练过程？
使用 `plot_training_results.py` 脚本可视化训练曲线和指标。

## 📝 更新日志

- **v1.0**: 初始版本，支持标准U-Net
- **v2.0**: 新增MobileNetV3-U-Net和VGG16-U-Net变体
- **v2.1**: 优化README文档，完善使用说明

## 📄 许可证

本项目参考的开源仓库遵循其各自的许可证。

---

**提示**：如有问题或建议，欢迎提交Issue或Pull Request！
