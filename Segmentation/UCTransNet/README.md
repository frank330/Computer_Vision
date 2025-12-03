# UCTransNet 图像分割项目

## 📋 项目简介

UCTransNet 是一个基于 Transformer 增强的 U-Net 架构的深度学习模型，专门用于医学图像分割任务。该项目结合了 U-Net 的编码-解码结构和 Transformer 的注意力机制，能够更准确地识别和分割图像中的目标区域。

### ✨ 主要特点

- **🔹 Transformer 增强**：使用 CTrans 模块捕获长距离依赖关系，提升特征表达能力
- **🔹 多尺度特征融合**：结合 U-Net 的编码-解码结构，保留细节信息
- **🔹 通道注意力机制**：通过 CCA 模块提升特征表达能力
- **🔹 高精度分割**：在医学图像分割任务上表现优异
- **🔹 Web 服务支持**：提供基于 Flask 的 Web 界面，方便使用
- **🔹 完整的训练流程**：支持训练、验证、测试和单张预测
- **🔹 TensorBoard 可视化**：实时监控训练过程和指标变化

---

## 📁 项目结构

```
UCTransNet/
├── Config.py                 # 配置文件，包含所有训练和模型参数
├── train_model.py            # 模型训练主程序
├── Train_one_epoch.py        # 单轮训练/验证实现
├── eval_model.py             # 模型测试评估脚本（在测试集上评估）
├── predict.py                # 单张图像预测脚本
├── web.py                    # Flask Web 服务
├── Load_Dataset.py           # 数据集加载和数据增强模块
├── preprocess_utils.py       # 图像预处理工具（确保训练和预测一致性）
├── utils.py                  # 工具函数（损失函数、评估指标、学习率调度器等）
├── nets/                     # 模型网络定义目录
│   ├── UCTransNet.py         # UCTransNet 主模型
│   └── CTrans.py             # Channel Transformer 模块
├── data/                     # 数据集目录
│   ├── training/             # 训练集
│   │   ├── images/           # 训练图像
│   │   └── labels/           # 训练标签
│   ├── val/                  # 验证集
│   │   ├── images/           # 验证图像
│   │   └── labels/           # 验证标签
│   └── test/                 # 测试集
│       ├── images/           # 测试图像
│       └── labels/           # 测试标签
├── log/                      # 日志和模型保存目录
│   ├── models/               # 保存的模型权重
│   │   └── best_model.pth.tar  # 最佳模型
│   ├── tensorboard_logs/     # TensorBoard 日志
│   ├── visualize_val/        # 验证结果可视化（按epoch保存）
│   └── *.log                 # 训练日志文件
├── templates/                # Web 服务模板目录
│   └── index.html            # Web 界面 HTML
└── static/                   # Web 服务静态文件

```

---

## 📄 文件详细说明

### 🔧 核心配置文件

#### `Config.py`
**作用**：项目配置文件，包含所有训练、验证和模型相关的参数设置。

**主要配置项**：
- **训练参数**：学习率、批次大小、训练轮数、图像尺寸等
- **模型参数**：输入通道数、输出类别数、模型名称等
- **数据路径**：训练集、验证集、测试集路径
- **输出路径**：模型保存路径、日志路径、可视化路径
- **Transformer 配置**：CTrans 模块的参数（头数、层数、patch 尺寸等）

**使用方法**：
```python
import Config as config

# 访问配置参数
print(config.learning_rate)  # 1e-3
print(config.batch_size)     # 4
print(config.img_size)       # 224

# 获取 Transformer 配置
config_vit = config.get_CTranS_config()
```

**重要配置参数**：
| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `learning_rate` | 初始学习率[train_model.py](train_model.py) | 1e-3 | 1e-3 ~ 1e-4 |
| `batch_size` | 批次大小 | 4 | 根据GPU内存调整 |
| `epochs` | 训练轮数 | 60 | 50 ~ 200 |
| `img_size` | 输入图像尺寸 | 224 | 224, 256, 512 |
| `n_channels` | 输入通道数 | 3 | 3 (RGB) |
| `n_labels` | 输出类别数 | 1 | 1 (二分类) |
| `cosineLR` | 是否使用余弦学习率 | True | True |
| `early_stopping_patience` | 早停耐心值 | 10 | 10 ~ 50 |

---

### 🧠 模型定义文件

#### `nets/UCTransNet.py`
**作用**：定义 UCTransNet 主模型架构。

**主要组件**：
- `ConvBatchNorm`：卷积 + 批归一化 + 激活函数
- `DownBlock`：下采样块（最大池化 + 卷积）
- `UpBlock_attention`：上采样块（带通道注意力机制）
- `CCA`：通道注意力模块
- `UCTransNet`：主模型类

**模型结构**：
1. **编码器**：4 个下采样层提取多尺度特征（64→128→256→512→512通道）
2. **CTrans 模块**：Transformer 增强的特征融合（融合4个尺度的特征）
3. **解码器**：4 个上采样层恢复分辨率（512→256→128→64通道）
4. **输出层**：1x1卷积生成分割结果

**使用方法**：
```python
from nets.UCTransNet import UCTransNet
import Config as config

config_vit = config.get_CTranS_config()
model = UCTransNet(
    config_vit, 
    n_channels=3,      # RGB输入
    n_classes=1,       # 二分类
    img_size=224       # 输入尺寸
)
```

#### `nets/CTrans.py`
**作用**：实现 Channel Transformer 模块，用于增强特征提取。

**主要组件**：
- `Channel_Embeddings`：通道嵌入层（将特征图转换为patch embeddings）
- `Attention_org`：多头注意力机制
- `Block_ViT`：Transformer 块（注意力 + MLP + 残差连接）
- `Encoder`：Transformer 编码器（多个Transformer块堆叠）
- `ChannelTransformer`：完整的通道 Transformer 模块

**功能**：
- 将不同尺度的特征图转换为 patch embeddings
- 使用多头注意力机制融合多尺度特征
- 通过残差连接保留原始特征信息
- 增强特征表达能力

---

### 📊 数据处理文件

#### `Load_Dataset.py`
**作用**：数据集加载和数据增强模块。

**主要功能**：

1. **数据增强函数**：
   - `random_rot_flip(image, label)`：随机旋转（0/90/180/270度）和翻转
   - `random_rotate(image, label)`：随机角度旋转（-20到20度）

2. **数据生成器**：
   - `RandomGenerator`：训练集数据增强生成器（随机旋转、翻转、缩放）
   - `ValGenerator`：验证集数据生成器（仅缩放，无随机增强）

3. **数据集类**：
   - `ImageToImage2D`：图像到图像的数据集类
     - 自动加载图像和标签
     - 支持数据增强
     - 自动调整图像尺寸
     - 标签二值化处理

**使用方法**：
```python
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader

# 训练集（带数据增强）
train_tf = RandomGenerator(output_size=[224, 224])
train_dataset = ImageToImage2D('./data/training/', train_tf, image_size=224)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 验证集（无随机增强）
val_tf = ValGenerator(output_size=[224, 224])
val_dataset = ImageToImage2D('./data/val/', val_tf, image_size=224)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
```

**数据集目录结构要求**：
```
data/
├── training/
│   ├── images/    # 训练图像（.jpg格式）
│   └── labels/    # 训练标签（.jpg格式，与图像同名）
├── val/
│   ├── images/    # 验证图像
│   └── labels/    # 验证标签
└── test/
    ├── images/    # 测试图像
    └── labels/    # 测试标签
```

**标签格式要求**：
- 标签图像应为二值化图像
- 背景像素值 ≤ 0，前景像素值 > 0
- 训练时会自动将标签转换为 0 和 1

#### `preprocess_utils.py`
**作用**：图像预处理工具模块，确保训练和预测时使用完全一致的预处理流程。

**主要功能**：
- `correct_dims(*images)`：修正图像维度（确保有正确的通道数）
- `preprocess_image_for_training(image_path, target_size)`：训练时的预处理
- `preprocess_image_for_prediction(image_path, target_size)`：预测时的预处理

**重要提示**：
- 该模块确保训练和预测使用相同的预处理步骤
- 使用 OpenCV 读取图像（BGR格式），然后转换为RGB
- 使用相同的resize方法和归一化方式

---

### 🚀 训练相关文件

#### `train_model.py`
**作用**：模型训练主程序，负责整个训练流程的管理。

**主要功能**：
- ✅ 初始化模型、优化器、损失函数
- ✅ 加载训练集和验证集
- ✅ 执行训练循环（多个 epoch）
- ✅ 保存最佳模型（基于验证集Dice系数）
- ✅ 记录 TensorBoard 日志（每轮平均指标）
- ✅ 实现早停机制（防止过拟合）
- ✅ 日志记录（训练过程详细记录）

**训练流程**：
1. 设置随机种子，确保可复现性
2. 创建数据加载器（训练集和验证集）
3. 初始化模型、优化器、损失函数
4. 初始化 TensorBoard（如果启用）
5. 循环训练多个 epoch：
   - 训练一个 epoch
   - 验证一个 epoch
   - 记录 TensorBoard 指标
   - 保存最佳模型
   - 检查早停条件
6. 训练结束，返回训练好的模型

**使用方法**：
```bash
python train_model.py
```

**输出文件**：
- **模型权重**：`./log/models/best_model.pth.tar`
- **训练日志**：`./log/train.log` 或 `./log/Test_session_*.log`
- **TensorBoard日志**：`./log/tensorboard_logs/`
- **验证可视化**：`./log/visualize_val/{epoch}/`

**TensorBoard 查看**：
```bash
tensorboard --logdir=./log/tensorboard_logs
```
然后在浏览器中访问 `http://localhost:6006`

**TensorBoard 记录的内容**：
- **Batch级别**：每个batch的损失、Dice、IoU
- **Epoch级别**：每轮的平均损失、Dice、IoU、学习率

#### `Train_one_epoch.py`
**作用**：实现单轮训练/验证的完整流程。

**主要功能**：
- 前向传播和损失计算
- 反向传播和参数更新（仅训练模式）
- 计算评估指标（Dice、IoU）
- 记录日志和 TensorBoard 信息
- 保存可视化结果（验证模式，按配置频率）
- 更新学习率（如果提供了学习率调度器）

**函数说明**：
```python
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger):
    """
    训练或验证一个完整的epoch
    
    Args:
        loader: 数据加载器
        model: 模型对象
        criterion: 损失函数
        optimizer: 优化器
        writer: TensorBoard写入器（可选）
        epoch: 当前epoch编号
        lr_scheduler: 学习率调度器（可选）
        logger: 日志记录器
    
    Returns:
        tuple: (average_loss, train_dice_avg, train_iou_average)
    """
```

**输出示例**：
```
[Train] Epoch: [1][10/403]  Loss:0.523 (Avg 0.6124) Dice:0.7234 (Avg 0.6891) LR 1.00e-03   (AvgTime 0.5)
```

---

### 🧪 测试和预测文件

#### `eval_model.py`
**作用**：模型测试评估脚本，用于在测试集上评估模型性能。

**主要功能**：
- ✅ 加载训练好的模型
- ✅ 在测试集上进行推理
- ✅ 计算每张图像的 Dice 系数
- ✅ 输出详细的统计信息（平均值、标准差、最高值、最低值）
- ✅ 确保预处理流程与训练时完全一致

**使用方法**：
```bash
python eval_model.py
```

**输出示例**：
```
找到 196 张测试图像
image_001: dice=0.8234
image_002: dice=0.7891
...
==================================================
测试结果统计
==================================================
测试图像数量: 196
平均Dice系数: 0.8123
Dice系数标准差: 0.0456
最高Dice系数: 0.9234
最低Dice系数: 0.6789
==================================================
```

**注意事项**：
- 确保模型文件存在于 `./log/models/best_model.pth.tar`
- 确保测试集路径在 `Config.py` 中正确配置
- 函数名不以 `test_` 开头，避免被 pytest 误识别

#### `predict.py`
**作用**：单张图像预测脚本，用于对单张图像进行分割预测。

**主要功能**：
- ✅ 加载训练好的模型（自动查找模型路径）
- ✅ 对单张图像进行预处理（与训练时完全一致）
- ✅ 执行模型推理
- ✅ 生成分割结果（原图、mask、叠加图）
- ✅ 返回推理时间和预测结果

**函数说明**：
```python
def predict_single_image(image_path, output_dir="./templates", save_results=True):
    """
    对单张图像进行分割预测
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录（默认 "./templates"）
        save_results: 是否保存结果图像（默认 True）
    
    Returns:
        dict: 包含预测结果信息的字典
            - original: 原图路径
            - mask: 分割mask路径
            - overlay: 叠加图路径
            - inference_time: 推理时间（秒）
            - prediction: 预测结果数组（如果save_results=False）
    """
```

**使用方法**：

**方式1：命令行使用**
```bash
python predict.py
```

**方式2：在代码中调用**
```python
from predict import predict_single_image

# 预测并保存结果
result = predict_single_image(
    image_path='./data/test/images/test.jpg',
    output_dir='./predict_results',
    save_results=True
)

print(f"推理时间: {result['inference_time']:.4f}秒")
print(f"原图: {result['original']}")
print(f"分割结果: {result['mask']}")
print(f"叠加图: {result['overlay']}")
```

**输出文件**：
- `{原文件名}_original.jpg` - 原始输入图像
- `{原文件名}_mask.jpg` - 分割结果（黑白mask）
- `{原文件名}_overlay.jpg` - 叠加可视化（原图+红色高亮分割区域）

**模型路径查找顺序**：
1. 使用 `config.test_session` 指定的路径
2. 尝试 `./log/models/best_model.pth.tar`
3. 自动查找最新的训练会话模型

---

### 🌐 Web 服务文件

#### `web.py`
**作用**：基于 Flask 的 Web 服务，提供图像分割的 Web 界面。

**主要功能**：
- ✅ 提供图像上传接口（支持 JPG、JPEG、PNG、BMP 格式）
- ✅ 调用预测函数进行分割
- ✅ 返回分割结果（原图、mask、叠加图）
- ✅ 提供图像访问接口
- ✅ 错误处理和日志记录

**路由说明**：
- `GET /`：主页面（显示上传界面）
- `POST /predict`：预测接口
  - 接收：multipart/form-data 格式的图像文件
  - 返回：JSON格式的预测结果
    ```json
    {
        "success": true,
        "message": "检测完成，推理时间: 0.123秒",
        "original": "/image/test_original.jpg",
        "mask": "/image/test_mask.jpg",
        "overlay": "/image/test_overlay.jpg",
        "inference_time": 0.123
    }
    ```
- `GET /image/<filename>`：图像访问接口（返回结果图像）

**使用方法**：
```bash
python web.py
```

然后访问 `http://localhost:5000`

**注意事项**：
- 确保模型文件存在于 `./log/models/best_model.pth.tar`
- 输出目录为 `./templates/`
- 默认端口为 5000，可在代码中修改

#### `templates/index.html`
**作用**：Web 服务的用户界面。

**主要功能**：
- ✅ 图像上传界面（拖拽或点击上传）
- ✅ 预测结果展示（原图、mask、叠加图三栏对比）
- ✅ 实时显示推理时间
- ✅ 响应式设计，支持移动端
- ✅ 美观的医疗主题UI设计

**界面特点**：
- 医疗蓝色主题
- 三栏对比展示（原图、分割结果、叠加显示）
- 实时进度提示
- 错误信息友好提示

---

### 🛠️ 工具函数文件

#### `utils.py`
**作用**：提供训练和评估过程中使用的各种工具函数。

**主要功能**：

1. **损失函数**：
   - `WeightedBCE`：加权二值交叉熵损失
     - 对正负样本分别设置权重
     - 适用于类别不平衡的数据集
   - `WeightedDiceLoss`：加权 Dice 损失
     - 关注重叠区域
     - 对边界敏感
   - `WeightedDiceBCE`：组合损失（Dice + BCE）
     - 结合两种损失的优点
     - 可调整权重平衡

2. **评估指标函数**：
   - `dice_coef(y_true, y_pred)`：计算 Dice 系数
   - `dice_on_batch(masks, pred)`：计算 batch 的平均 Dice 系数
   - `iou_on_batch(masks, pred)`：计算 batch 的平均 IoU
   - `auc_on_batch(masks, pred)`：计算 batch 的平均 AUC

3. **可视化函数**：
   - `save_on_batch(images, masks, pred, names, vis_path)`：保存 batch 的可视化结果

4. **学习率调度器**：
   - `CosineAnnealingWarmRestarts`：余弦退火重启学习率调度器（SGDR）
     - 周期性重启学习率
     - 帮助跳出局部最优
     - 支持周期长度倍增

**使用方法**：
```python
from utils import WeightedDiceBCE, dice_on_batch, CosineAnnealingWarmRestarts

# 创建损失函数
criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

# 计算评估指标
dice_score = dice_on_batch(masks, predictions)

# 创建学习率调度器
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,      # 第一个周期长度
    T_mult=1,    # 周期倍增因子
    eta_min=1e-4 # 最小学习率
)
```

---

## 🚀 快速开始

### 1. 环境要求

```bash
Python >= 3.7
PyTorch >= 1.7.0
CUDA >= 10.2 (如果使用GPU)
```

### 2. 安装依赖

```bash
# 安装PyTorch（根据你的CUDA版本选择）
pip install torch torchvision

# 安装其他依赖
pip install numpy opencv-python Pillow scikit-learn
pip install tensorboardX flask flask-cors ml-collections tqdm
```

### 3. 数据准备

**数据集结构**：
```
data/
├── training/
│   ├── images/    # 训练图像（.jpg格式）
│   └── labels/    # 训练标签（.jpg格式，与图像同名）
├── val/
│   ├── images/    # 验证图像
│   └── labels/    # 验证标签
└── test/
    ├── images/    # 测试图像
    └── labels/    # 测试标签
```

**图像要求**：
- 图像和标签文件名必须一致（扩展名可以不同）
- 标签图像应为二值化图像（背景≤0，前景>0）
- 建议图像尺寸统一（训练时会自动调整到 `config.img_size`）

### 4. 配置参数

编辑 `Config.py`，设置：
- 数据集路径（`train_dataset`、`val_dataset`、`test_dataset`）
- 训练参数（`learning_rate`、`batch_size`、`epochs`等）
- 模型参数（`n_channels`、`n_labels`、`img_size`等）

### 5. 训练模型

```bash
python train_model.py
```

**训练过程**：
- 训练日志会实时输出到控制台
- 同时保存到 `./log/*.log` 文件
- TensorBoard 日志保存在 `./log/tensorboard_logs/`
- 最佳模型保存在 `./log/models/best_model.pth.tar`

**监控训练**：
```bash
# 启动TensorBoard
tensorboard --logdir=./log/tensorboard_logs

# 然后在浏览器访问 http://localhost:6006
```

### 6. 测试模型

```bash
python eval_model.py
```

### 7. 单张图像预测

```python
from predict import predict_single_image

result = predict_single_image('./data/test/images/test.jpg')
print(f"推理时间: {result['inference_time']:.4f}秒")
```

### 8. 启动 Web 服务

```bash
python web.py
```

然后访问 `http://localhost:5000`，上传图像进行分割。

---

## 📊 评估指标说明

### Dice 系数（Dice Coefficient）
衡量预测结果和真实标签的重叠程度：
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```
- **范围**：[0, 1]，越大越好
- **特点**：对重叠区域敏感，适合评估分割质量

### IoU（Intersection over Union）
交并比，衡量预测和真实标签的重叠区域：
```
IoU = |A ∩ B| / |A ∪ B|
```
- **范围**：[0, 1]，越大越好
- **特点**：同时考虑交集和并集，更严格的评估指标

### AUC（Area Under Curve）
ROC 曲线下面积，衡量分类性能：
- **范围**：[0, 1]，越大越好
- **特点**：评估模型区分正负样本的能力

---

## ⚙️ 配置参数详解

### 训练参数（Config.py）

| 参数 | 说明 | 默认值 | 调整建议 |
|------|------|--------|----------|
| `learning_rate` | 初始学习率 | 1e-3 | 1e-3 ~ 1e-4，根据收敛情况调整 |
| `batch_size` | 批次大小 | 4 | 根据GPU内存调整（4/8/16） |
| `epochs` | 训练轮数 | 60 | 50 ~ 200，配合早停使用 |
| `img_size` | 输入图像尺寸 | 224 | 224/256/512，越大精度可能越高但速度越慢 |
| `n_channels` | 输入通道数 | 3 | 3 (RGB) |
| `n_labels` | 输出类别数 | 1 | 1 (二分类) |
| `cosineLR` | 是否使用余弦学习率 | True | 推荐使用 |
| `early_stopping_patience` | 早停耐心值 | 10 | 10 ~ 50，防止过早停止 |
| `print_frequency` | 打印频率 | 10 | 每N个batch打印一次 |
| `vis_frequency` | 可视化频率 | 10 | 每N个epoch保存一次可视化结果 |

### Transformer 参数

| 参数 | 说明 | 默认值 | 调整建议 |
|------|------|--------|----------|
| `num_heads` | 多头注意力头数 | 4 | 4/8，必须是KV_size的因子 |
| `num_layers` | Transformer 层数 | 4 | 2 ~ 6，层数越多表达能力越强但计算量越大 |
| `expand_ratio` | MLP 扩展比例 | 4 | 2 ~ 8 |
| `patch_sizes` | Patch 尺寸列表 | [16,8,4,2] | 对应4个尺度的特征图 |
| `KV_size` | Key-Value 尺寸 | 960 | 由模型结构决定，一般不需要修改 |
| `base_channel` | U-Net 基础通道数 | 64 | 32/64/128，影响模型大小和精度 |

---

## ❓ 常见问题

### 1. 内存不足（Out of Memory）

**解决方案**：
- 减小 `batch_size`（如从4改为2或1）
- 减小 `img_size`（如从224改为128）
- 使用梯度累积（修改训练代码）
- 清理GPU缓存（代码中已包含 `torch.cuda.empty_cache()`）

### 2. 训练速度慢

**解决方案**：
- 使用 GPU 训练（确保 `CUDA_VISIBLE_DEVICES` 正确设置）
- 增加 `num_workers`（如果数据加载是瓶颈，当前为0可改为2-4）
- 减小 `img_size` 或模型复杂度
- 使用混合精度训练（需要修改代码）

### 3. 模型不收敛

**解决方案**：
- 检查学习率是否合适（尝试更小的学习率，如1e-4）
- 检查数据标签是否正确（确保标签是二值化的）
- 尝试不同的损失函数权重（调整 `dice_weight` 和 `BCE_weight`）
- 增加训练轮数
- 检查数据增强是否过于激进

### 4. 验证指标不提升

**解决方案**：
- 检查验证集是否正确（确保验证集有标签）
- 调整早停耐心值（增加 `early_stopping_patience`）
- 尝试不同的数据增强策略
- 检查模型是否过拟合（训练集指标远高于验证集）
- 尝试正则化（Dropout、权重衰减）

### 5. 单张预测效果差（训练/测试效果好但预测效果差）

**问题原因**：
- 预处理流程不一致：训练时使用 OpenCV 读取图像（BGR），预测时可能使用 PIL（RGB）
- 归一化参数不一致：训练和预测时使用了不同的归一化方式
- 模型状态未正确切换：未使用 `model.eval()` 关闭 Dropout 和 BatchNorm 的训练模式

**解决方案**：
1. **使用统一的预处理函数**：
   - 使用 `predict.py` 中的 `predict_single_image()` 函数
   - 或使用 `preprocess_utils.py` 中的预处理函数

2. **确保模型处于评估模式**：
   ```python
   model.eval()  # 关闭Dropout和BatchNorm的训练模式
   with torch.no_grad():  # 禁用梯度计算
       output = model(input_tensor)
   ```

3. **检查输入格式**：
   - 确保输入维度为 `[1, C, H, W]`（batch维度为1）
   - 确保图像值在 `[0, 1]` 范围内（已归一化）
   - 确保图像尺寸与训练时一致

4. **验证预处理一致性**：
   - 对比训练时的预处理代码（`Load_Dataset.py`）和预测时的预处理代码（`predict.py`）
   - 确保使用相同的图像读取方式、resize方法和归一化方式

### 6. TensorBoard 无法显示数据

**解决方案**：
- 检查 `config.tensorboard` 是否为 `True`
- 检查日志目录路径是否正确
- 确保 TensorBoard 版本兼容
- 尝试使用 `tensorboard --logdir=./log/tensorboard_logs --reload_interval=1`

### 7. 模型加载失败

**解决方案**：
- 检查模型文件路径是否正确
- 检查模型文件是否完整（文件大小是否正常）
- 确保模型架构与保存时一致
- 检查 PyTorch 版本是否兼容

---

## 💡 使用技巧

### 1. 训练技巧

- **学习率调整**：如果训练初期损失不下降，尝试减小学习率
- **早停策略**：设置合适的 `early_stopping_patience`，避免过拟合
- **数据增强**：适当的数据增强可以提高模型泛化能力
- **模型保存**：只保存验证集上表现最好的模型

### 2. 预测技巧

- **批量预测**：对于多张图像，可以修改代码实现批量预测，提高效率
- **结果后处理**：可以对预测结果进行形态学操作（开运算、闭运算）优化分割结果
- **阈值调整**：如果默认0.5阈值效果不好，可以尝试调整阈值

### 3. Web 服务部署

- **生产环境**：使用 Gunicorn 或 uWSGI 部署，不要直接使用 Flask 开发服务器
- **安全考虑**：添加文件大小限制、文件类型检查、请求频率限制
- **性能优化**：使用模型量化、ONNX 转换等方式加速推理

---

## 🔬 项目改进建议

### 短期改进

1. **数据增强**：
   - 添加更多数据增强方法（弹性变形、颜色抖动、MixUp等）
   - 实现在线数据增强

2. **训练策略**：
   - 实现混合精度训练（FP16）
   - 添加学习率查找器
   - 实现模型检查点恢复（断点续训）

3. **评估指标**：
   - 添加更多评估指标（Hausdorff距离、95%HD等）
   - 添加混淆矩阵分析
   - 实现ROC曲线绘制

### 长期改进

1. **模型优化**：
   - 尝试不同的 Transformer 配置
   - 实现模型集成（Ensemble）
   - 添加测试时增强（TTA）

2. **部署优化**：
   - 模型量化（INT8）
   - 模型剪枝
   - ONNX 导出
   - TensorRT 加速

3. **功能扩展**：
   - 支持多类别分割
   - 支持3D图像分割
   - 添加模型解释性分析（Grad-CAM等）

---

## 📚 参考文献

- **UCTransNet**: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer
- **U-Net**: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
- **Transformer**: Attention Is All You Need (NIPS 2017)
- **SGDR**: Stochastic Gradient Descent with Warm Restarts (ICLR 2017)

---

## 📝 更新日志

### v1.0 (当前版本)
- ✅ 完整的训练、验证、测试流程
- ✅ TensorBoard 可视化支持
- ✅ Web 服务界面
- ✅ 单张图像预测功能
- ✅ 详细的中文注释
- ✅ 完善的错误处理

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进项目！

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**：2024年
