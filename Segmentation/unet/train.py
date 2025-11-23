"""
U-Net 图像分割模型训练脚本

该脚本用于训练U-Net模型进行图像分割任务，支持：
- 单GPU/CPU训练
- 混合精度训练（AMP）
- 断点续训
- 自动保存最佳模型
- 训练过程指标记录

使用方法：
    1. 在 TRAIN_CONFIG 中直接修改参数
    2. 运行: python train.py
    3. 或使用命令行参数: python train.py --epochs 100 --batch-size 8
"""

import os          # 文件系统操作
import time        # 时间相关操作，用于计算训练时间
import datetime    # 日期时间处理，用于生成结果文件名

import torch       # PyTorch深度学习框架

# 导入U-Net模型
from src import UNet,MobileV3Unet,VGG16UNet
# 导入训练工具函数：训练一个epoch、评估模型、创建学习率调度器
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
# 导入自定义数据集类
from my_dataset import MyDataset
# 导入数据变换模块
import transforms as T

# ============================================================================
# 训练参数配置区域 - 在这里直接修改参数即可
# ============================================================================
TRAIN_CONFIG = {
    "data_path": "./",              # 数据集根目录路径
    "num_classes": 1,               # 类别数（不包括背景）
    "device": "cuda",               # 训练设备: "cuda" 或 "cpu"
    "batch_size": 4,                # 批次大小
    "epochs": 20,                  # 训练轮数
    "lr": 0.01,                     # 初始学习率
    "momentum": 0.9,                # 动量
    "weight_decay": 1e-4,           # 权重衰减
    "print_freq": 1,                # 打印频率
    "resume": "",                   # 恢复训练的检查点路径（留空则不恢复）
    "start_epoch": 1,               # 起始轮数
    "save_best": True,              # 是否只保存最佳模型
    "amp": False,                   # 是否使用混合精度训练
}
# ============================================================================


class SegmentationPresetTrain:
    """
    训练时的数据增强预设类
    
    该类定义了训练时对图像和标签进行的数据增强操作，包括：
    - 随机尺寸调整
    - 随机水平/垂直翻转
    - 随机裁剪
    - 转换为Tensor
    - 标准化
    """
    
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        初始化训练数据增强预设
        
        Args:
            base_size (int): 基础图像尺寸，用于计算随机调整的范围
            crop_size (int): 随机裁剪后的目标尺寸
            hflip_prob (float): 水平翻转概率，范围[0, 1]，默认0.5
            vflip_prob (float): 垂直翻转概率，范围[0, 1]，默认0.5
            mean (tuple): 图像各通道的均值，用于标准化，默认ImageNet均值
            std (tuple): 图像各通道的标准差，用于标准化，默认ImageNet标准差
        """
        # 计算随机调整尺寸的范围：最小为base_size的50%，最大为base_size的120%
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        # 构建数据增强变换序列
        trans = [T.RandomResize(min_size, max_size)]  # 随机尺寸调整
        
        # 根据概率决定是否添加翻转操作
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))  # 随机水平翻转
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))   # 随机垂直翻转
        
        # 添加其他必要的变换
        trans.extend([
            T.RandomCrop(crop_size),    # 随机裁剪到固定尺寸
            T.ToTensor(),                # 转换为PyTorch Tensor
            T.Normalize(mean=mean, std=std),  # 标准化（归一化）
        ])
        
        # 组合所有变换
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        """
        对图像和标签应用数据增强变换
        
        Args:
            img: PIL Image，输入图像
            target: PIL Image，分割标签
            
        Returns:
            tuple: (变换后的图像tensor, 变换后的标签tensor)
        """
        return self.transforms(img, target)


class SegmentationPresetEval:
    """
    验证/测试时的数据预处理预设类
    
    验证时不需要数据增强，只需要：
    - 转换为Tensor
    - 标准化
    """
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        初始化验证数据预处理预设
        
        Args:
            mean (tuple): 图像各通道的均值，用于标准化，默认ImageNet均值
            std (tuple): 图像各通道的标准差，用于标准化，默认ImageNet标准差
        """
        # 验证时只进行必要的预处理，不进行数据增强
        self.transforms = T.Compose([
            T.ToTensor(),                # 转换为PyTorch Tensor
            T.Normalize(mean=mean, std=std),  # 标准化
        ])

    def __call__(self, img, target):
        """
        对图像和标签应用预处理变换
        
        Args:
            img: PIL Image，输入图像
            target: PIL Image，分割标签
            
        Returns:
            tuple: (变换后的图像tensor, 变换后的标签tensor)
        """
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    根据训练/验证模式获取相应的数据变换
    
    Args:
        train (bool): True表示训练模式（使用数据增强），False表示验证模式（不使用数据增强）
        mean (tuple): 图像各通道的均值，用于标准化
        std (tuple): 图像各通道的标准差，用于标准化
        
    Returns:
        SegmentationPresetTrain 或 SegmentationPresetEval: 相应的数据变换对象
    """
    # 定义图像尺寸参数
    base_size = 565  # 基础图像尺寸，用于计算随机调整范围
    crop_size = 480  # 裁剪后的目标尺寸

    if train:
        # 训练模式：返回包含数据增强的变换
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        # 验证模式：返回只包含预处理的变换
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    """
    创建U-Net模型实例
    
    Args:
        num_classes (int): 分割类别数（包括背景）
            例如：二分类任务（背景+前景）num_classes=2
        
    Returns:
        UNet: U-Net模型实例
    """
    # 创建U-Net模型
    # in_channels=3: 输入图像通道数（RGB图像）
    # num_classes: 输出类别数（包括背景）

    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    # model = MobileV3Unet( num_classes=num_classes, )
    # model = VGG16UNet(num_classes=num_classes, )

    return model


def main(args):
    """
    主训练函数
    
    该函数负责：
    1. 初始化训练环境（设备、数据集、模型等）
    2. 配置优化器和学习率调度器
    3. 执行训练循环
    4. 保存模型和训练结果
    
    Args:
        args: 命令行参数对象，包含所有训练配置参数
    """
    # ==================== 1. 设备配置 ====================
    # 根据CUDA可用性和用户设置选择训练设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    
    # ==================== 2. 类别数配置 ====================
    # 分割任务中，类别数 = 前景类别数 + 背景（1个）
    # 例如：二分类任务（背景+前景）num_classes = 1 + 1 = 2
    num_classes = args.num_classes + 1

    # ==================== 3. 数据标准化参数 ====================
    # 这些均值和标准差应该使用 compute_mean_std.py 脚本计算得到
    # 用于对图像进行标准化，有助于模型训练稳定
    mean = (0.709, 0.381, 0.224)  # RGB三通道的均值
    std = (0.127, 0.079, 0.043)   # RGB三通道的标准差

    # ==================== 4. 结果文件配置 ====================
    # 生成带时间戳的结果文件名，用于保存训练过程中的指标
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # ==================== 5. 数据集初始化 ====================
    # 创建训练集，使用数据增强
    train_dataset = MyDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    # 创建验证集，不使用数据增强
    val_dataset = MyDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    # ==================== 6. 数据加载器配置 ====================
    # 计算数据加载的工作进程数
    # 限制条件：不超过CPU核心数、不超过batch_size、最大为8
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    
    # 训练数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,      # 批次大小
        num_workers=num_workers,     # 数据加载的工作进程数
        shuffle=True,                # 训练时打乱数据顺序
        pin_memory=True,             # 将数据固定在内存中，加速GPU传输
        collate_fn=train_dataset.collate_fn  # 自定义批处理函数，处理不同尺寸的图像
    )

    # 验证数据加载器
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,                # 验证时使用batch_size=1，便于处理不同尺寸的图像
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    # ==================== 7. 模型初始化 ====================
    model = create_model(num_classes=num_classes)
    model.to(device)  # 将模型移动到指定设备（GPU或CPU）

    # ==================== 8. 优化器配置 ====================
    # 获取所有需要训练的参数（requires_grad=True的参数）
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 使用SGD优化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,                    # 初始学习率
        momentum=args.momentum,         # 动量系数，有助于加速训练和稳定收敛
        weight_decay=args.weight_decay  # 权重衰减（L2正则化），防止过拟合
    )

    # ==================== 9. 混合精度训练配置 ====================
    # 如果启用混合精度训练（AMP），可以加速训练并减少显存占用
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # ==================== 10. 学习率调度器配置 ====================
    # 创建学习率更新策略
    # 注意：这里是每个step更新一次学习率（不是每个epoch）
    # warmup=True: 启用学习率预热，训练初期逐渐增加学习率
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # ==================== 11. 断点续训 ====================
    # 如果指定了检查点路径，则加载之前的训练状态继续训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])              # 加载模型权重
        optimizer.load_state_dict(checkpoint['optimizer'])      # 加载优化器状态
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  # 加载学习率调度器状态
        args.start_epoch = checkpoint['epoch'] + 1               # 从下一个epoch开始训练
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])         # 加载混合精度训练的状态

    # ==================== 12. 训练循环 ====================
    best_dice = 0.  # 记录最佳Dice系数
    start_time = time.time()  # 记录训练开始时间
    
    for epoch in range(args.start_epoch, args.epochs):
        # ---------- 12.1 训练一个epoch ----------
        # 返回该epoch的平均损失和学习率
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch, num_classes,
            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler
        )

        # ---------- 12.2 在验证集上评估模型 ----------
        # 返回混淆矩阵和Dice系数
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)  # 将混淆矩阵转换为字符串
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        
        # ---------- 12.3 保存训练指标到文件 ----------
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        # ---------- 12.4 保存模型检查点 ----------
        # 如果启用只保存最佳模型，则只在Dice系数提升时保存
        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice  # 更新最佳Dice系数
            else:
                continue  # 如果Dice系数没有提升，跳过保存

        # 准备保存的检查点内容
        save_file = {
            "model": model.state_dict(),                    # 模型权重
            "optimizer": optimizer.state_dict(),            # 优化器状态
            "lr_scheduler": lr_scheduler.state_dict(),      # 学习率调度器状态
            "epoch": epoch,                                 # 当前epoch数
            "args": args                                    # 训练参数（用于恢复训练）
        }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()      # 混合精度训练的状态

        # 根据配置决定保存方式
        if args.save_best is True:
            # 只保存最佳模型（覆盖之前的best_model.pth）
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            # 保存每个epoch的模型
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    # ==================== 13. 训练完成 ====================
    # 计算总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    """
    解析命令行参数
    
    该函数解析命令行传入的训练参数。如果命令行没有指定某个参数，
    则使用 TRAIN_CONFIG 中的默认配置。
    
    注意：命令行参数的优先级高于代码中的 TRAIN_CONFIG 配置
    
    Returns:
        argparse.Namespace: 包含所有训练参数的对象
    """
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    # ========== 数据集相关参数 ==========
    parser.add_argument("--data-path", default=TRAIN_CONFIG["data_path"], 
                       help="数据集根目录路径")
    
    # ========== 模型相关参数 ==========
    # 注意：num_classes不包括背景，实际模型输出类别数 = num_classes + 1
    parser.add_argument("--num-classes", default=TRAIN_CONFIG["num_classes"], type=int,
                       help="分割类别数（不包括背景）")
    
    # ========== 训练设备参数 ==========
    parser.add_argument("--device", default=TRAIN_CONFIG["device"], 
                       help="训练设备: 'cuda' 或 'cpu'")
    
    # ========== 训练超参数 ==========
    parser.add_argument("-b", "--batch-size", default=TRAIN_CONFIG["batch_size"], type=int,
                       help="批次大小（batch size）")
    parser.add_argument("--epochs", default=TRAIN_CONFIG["epochs"], type=int, metavar="N",
                       help="总训练轮数")
    parser.add_argument('--lr', default=TRAIN_CONFIG["lr"], type=float, 
                       help='初始学习率（learning rate）')
    parser.add_argument('--momentum', default=TRAIN_CONFIG["momentum"], type=float, metavar='M',
                       help='SGD优化器的动量系数，范围[0, 1]')
    parser.add_argument('--wd', '--weight-decay', default=TRAIN_CONFIG["weight_decay"], type=float,
                       metavar='W', help='权重衰减系数（L2正则化），默认1e-4',
                       dest='weight_decay')
    
    # ========== 训练控制参数 ==========
    parser.add_argument('--print-freq', default=TRAIN_CONFIG["print_freq"], type=int, 
                       help='打印训练信息的频率（每N个batch打印一次）')
    parser.add_argument('--resume', default=TRAIN_CONFIG["resume"], 
                       help='恢复训练的检查点路径（留空则不恢复训练）')
    parser.add_argument('--start-epoch', default=TRAIN_CONFIG["start_epoch"], type=int, metavar='N',
                       help='起始训练轮数（通常用于断点续训）')
    parser.add_argument('--save-best', default=TRAIN_CONFIG["save_best"], type=bool, 
                       help='是否只保存最佳模型（True: 只保存Dice系数最高的模型，False: 保存每个epoch的模型）')
    
    # ========== 混合精度训练参数 ==========
    parser.add_argument("--amp", default=TRAIN_CONFIG["amp"], type=bool,
                       help="是否使用混合精度训练（Automatic Mixed Precision），可以加速训练并减少显存占用")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """
    主程序入口
    
    执行流程：
    1. 解析命令行参数（优先使用命令行参数，如果没有则使用 TRAIN_CONFIG 中的配置）
    2. 创建模型权重保存目录
    3. 调用main函数开始训练
    """
    # 解析参数（优先使用命令行参数，如果没有则使用 TRAIN_CONFIG 中的配置）
    args = parse_args()

    # 确保模型权重保存目录存在
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # 开始训练
    main(args)
