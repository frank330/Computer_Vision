"""
基于MobileNetV3的U-Net图像分割模型

该模块实现了使用MobileNetV3作为编码器（backbone）的U-Net分割网络。
MobileNetV3是一个轻量级的卷积神经网络，适合在移动设备或资源受限的环境中运行。

主要组件：
- IntermediateLayerGetter: 从backbone中提取中间层特征的包装器
- MobileV3Unet: 完整的MobileNetV3-U-Net分割模型
"""

from collections import OrderedDict  # 有序字典，保持模块顺序
from typing import Dict              # 类型提示：字典类型
import torch                         # PyTorch核心库
import torch.nn as nn                # PyTorch神经网络模块
import torch.nn.functional as F      # PyTorch函数式接口
from torch import Tensor             # 张量类型提示
from torchvision.models import mobilenet_v3_large  # MobileNetV3 Large预训练模型
from .unet import Up, OutConv         # 导入U-Net的上采样层和输出卷积层


class IntermediateLayerGetter(nn.ModuleDict):
    """
    中间层特征提取器
    
    该类是一个模块包装器，用于从模型中提取指定中间层的特征输出。
    这对于构建编码器-解码器架构（如U-Net）非常有用，因为需要从编码器的
    不同阶段获取特征图用于跳跃连接。
    
    重要假设：
    1. 模块必须按照使用顺序注册到模型中
    2. 同一个nn.Module不能在forward中重复使用两次
    3. 只能查询直接分配给模型的子模块，不能查询嵌套的子模块
       例如：可以返回 `model.feature1`，但不能返回 `model.feature1.layer2`
    
    Args:
        model (nn.Module): 要提取特征的模型
        return_layers (Dict[str, str]): 字典，键为模块名称，值为返回特征的名称
            例如：{"layer1": "stage0", "layer2": "stage1"} 表示从layer1提取特征
            并命名为stage0，从layer2提取特征并命名为stage1
    """
    _version = 2  # 版本号
    __annotations__ = {
        "return_layers": Dict[str, str],  # 类型注解：返回层字典
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        """
        初始化中间层特征提取器
        
        Args:
            model: 要提取特征的模型
            return_layers: 指定要返回的层及其新名称的字典
        """
        # 检查指定的层是否都存在于模型中
        model_layer_names = [name for name, _ in model.named_children()]
        if not set(return_layers.keys()).issubset(set(model_layer_names)):
            raise ValueError("return_layers中指定的层在模型中不存在")
        
        # 保存原始的return_layers（用于forward中查找）
        orig_return_layers = return_layers
        # 将键和值都转换为字符串类型，确保一致性
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，只保留需要的模块，删除不需要的模块以节省内存
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module  # 添加模块到新的有序字典
            if name in return_layers:
                # 如果这个模块是我们需要的，从return_layers中删除
                del return_layers[name]
            # 如果所有需要的层都已找到，提前结束循环
            if not return_layers:
                break

        # 调用父类构造函数，初始化ModuleDict
        super(IntermediateLayerGetter, self).__init__(layers)
        # 保存原始的return_layers映射（用于forward中查找）
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        前向传播，提取指定层的特征
        
        Args:
            x (Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            Dict[str, Tensor]: 包含指定层特征的字典，键为return_layers中指定的名称
        """
        out = OrderedDict()  # 用于存储提取的特征
        
        # 按顺序遍历所有模块
        for name, module in self.items():
            x = module(x)  # 通过模块进行前向传播
            # 如果当前模块在return_layers中，保存其输出
            if name in self.return_layers:
                out_name = self.return_layers[name]  # 获取指定的输出名称
                out[out_name] = x  # 保存特征图
        
        return out


class MobileV3Unet(nn.Module):
    """
    基于MobileNetV3的U-Net图像分割模型
    
    该模型使用MobileNetV3作为编码器（backbone），提取多尺度特征，
    然后通过U-Net的解码器结构进行上采样，最终输出分割结果。
    
    网络结构：
    1. 编码器（MobileNetV3）：提取5个不同尺度的特征图
    2. 解码器（U-Net上采样）：通过4个上采样层逐步恢复分辨率
    3. 输出层：生成最终的分割结果
    
    优势：
    - 轻量级：参数量少，适合移动设备
    - 高效：推理速度快
    - 准确：在保持轻量级的同时，仍能获得较好的分割精度
    """
    
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        """
        初始化MobileNetV3-U-Net模型
        
        Args:
            num_classes (int): 分割类别数（包括背景）
                例如：二分类任务（背景+前景）num_classes=2
            pretrain_backbone (bool): 是否使用ImageNet预训练的MobileNetV3权重
                True: 使用预训练权重，可以加速训练和提高精度
                False: 随机初始化权重
        """
        super(MobileV3Unet, self).__init__()
        
        # ==================== 1. 加载MobileNetV3 backbone ====================
        # 创建MobileNetV3 Large模型
        # pretrained参数：是否加载ImageNet预训练权重
        backbone = mobilenet_v3_large(pretrained=pretrain_backbone)

        # 如果需要手动加载预训练权重，可以使用以下代码（已注释）
        # if pretrain_backbone:
        #     # 载入mobilenetv3 large backbone预训练权重
        #     # 下载地址：https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
        #     backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

        # 只使用MobileNetV3的features部分（去掉分类头）
        backbone = backbone.features

        # ==================== 2. 定义要提取的特征层 ====================
        # stage_indices: 指定要提取的层索引
        # MobileNetV3的features包含16个模块，这里选择5个关键层用于跳跃连接
        # 索引对应：浅层（细节）-> 深层（语义）
        stage_indices = [1, 3, 6, 12, 15]
        
        # 获取每个阶段的输出通道数
        self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        
        # 创建return_layers字典，将层索引映射到stage名称
        # 例如：{1: "stage0", 3: "stage1", 6: "stage2", 12: "stage3", 15: "stage4"}
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        
        # 使用IntermediateLayerGetter包装backbone，提取指定层的特征
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # ==================== 3. 构建U-Net解码器 ====================
        # 解码器通过上采样层逐步恢复分辨率，并与编码器对应层的特征进行跳跃连接
        
        # 第一个上采样层：融合stage4和stage3的特征
        # 输入通道数 = stage4通道数 + stage3通道数
        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])  # 上采样并输出stage3通道数
        
        # 第二个上采样层：融合上一步输出和stage2的特征
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        
        # 第三个上采样层：融合上一步输出和stage1的特征
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        
        # 第四个上采样层：融合上一步输出和stage0的特征
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        
        # ==================== 4. 输出层 ====================
        # 最终输出层：将特征图转换为分割结果
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        网络流程：
        1. 通过MobileNetV3编码器提取多尺度特征
        2. 通过U-Net解码器逐步上采样，并与编码器特征进行跳跃连接
        3. 输出最终的分割结果
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, 3, H, W)
                B: batch size（批次大小）
                3: RGB三通道
                H, W: 图像高度和宽度
        
        Returns:
            Dict[str, torch.Tensor]: 包含分割结果的字典
                "out": 分割结果，形状为 (B, num_classes, H, W)
        """
        # 保存输入图像的尺寸，用于最后的上采样
        input_shape = x.shape[-2:]  # 获取高度和宽度 (H, W)
        
        # ==================== 编码器：提取多尺度特征 ====================
        # 通过backbone提取5个不同尺度的特征图
        backbone_out = self.backbone(x)
        # backbone_out包含：
        # - 'stage0': 最浅层特征（细节丰富，分辨率高）
        # - 'stage1': 较浅层特征
        # - 'stage2': 中间层特征
        # - 'stage3': 较深层特征
        # - 'stage4': 最深层特征（语义丰富，分辨率低）
        
        # ==================== 解码器：逐步上采样 ====================
        # 从最深层开始，逐步上采样并与对应层的编码器特征进行跳跃连接
        
        # 第一个上采样：融合最深层特征（stage4）和次深层特征（stage3）
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        
        # 第二个上采样：融合上一步输出和stage2的特征
        x = self.up2(x, backbone_out['stage2'])
        
        # 第三个上采样：融合上一步输出和stage1的特征
        x = self.up3(x, backbone_out['stage1'])
        
        # 第四个上采样：融合上一步输出和最浅层特征（stage0）
        x = self.up4(x, backbone_out['stage0'])
        
        # ==================== 输出层 ====================
        # 通过输出卷积层生成最终的分割结果
        x = self.conv(x)
        
        # ==================== 上采样到原始输入尺寸 ====================
        # 使用双线性插值将输出上采样到输入图像的尺寸
        # 这样可以处理任意尺寸的输入图像
        x = F.interpolate(
            x, 
            size=input_shape,           # 目标尺寸（输入图像的高度和宽度）
            mode="bilinear",           # 双线性插值
            align_corners=False         # 对齐方式
        )

        # 返回分割结果（字典格式，与标准U-Net输出格式一致）
        return {"out": x}
