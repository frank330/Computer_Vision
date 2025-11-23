"""
预测脚本：使用训练好的U-Net模型对单张图像进行分割预测

"""
import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    """
    同步CUDA操作并返回当前时间，用于精确测量推理时间
    
    Returns:
        float: 当前时间戳
    """
    # 如果使用GPU，等待所有CUDA操作完成以确保时间测量的准确性
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(path, output_dir="./templates"):
    """
    主函数：加载模型、图像，进行预测并保存结果
    
    Args:
        path: 输入图像路径
        output_dir: 输出目录，用于保存结果图像
        
    Returns:
        dict: 包含原图路径、分割mask路径和叠加图路径的字典
    """
    # 分割类别数（不包括背景），1表示只有前景（息肉）
    classes = 1
    
    # 模型权重文件路径
    weights_path = "./save_weights/best_model1.pth"
    
    # 检查文件是否存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(path), f"image {path} not found."

    # 数据集的均值和标准差（用于图像归一化）
    # 这些值应该与训练时使用的值保持一致
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 获取计算设备（优先使用GPU，如果没有则使用CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建U-Net模型
    # in_channels=3: RGB三通道输入
    # num_classes=classes+1: 类别数（背景+前景），即2类
    # base_c=32: 基础通道数
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # 加载训练好的模型权重
    # map_location='cpu': 先将权重加载到CPU，避免GPU内存问题
    # weights_only=False: 允许加载完整的checkpoint（包含model、optimizer等）
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False)['model'])
    model.to(device)

    # 加载待预测的图像并转换为RGB格式
    original_img = Image.open(path).convert('RGB')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义图像预处理变换：转换为tensor并归一化
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL Image转换为tensor，并归一化到[0,1]
        transforms.Normalize(mean=mean, std=std)  # 使用均值和标准差进行归一化
    ])
    img = data_transform(original_img)
    
    # 添加batch维度：[C, H, W] -> [1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # 将模型设置为评估模式（关闭dropout、batch normalization的更新等）
    model.eval()
    
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速推理
        # 模型预热：使用零张量进行一次前向传播
        # 这可以初始化模型中的一些层（如batch normalization），确保第一次推理的准确性
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        # 记录推理开始时间
        t_start = time_synchronized()
        
        # 进行模型推理
        output = model(img.to(device))
        
        # 记录推理结束时间并打印
        t_end = time_synchronized()
        print("inference time: {}s".format(t_end - t_start))

        # 获取预测结果
        # output['out']的形状为[1, num_classes, H, W]
        # argmax(1): 在类别维度上取最大值索引，得到预测的类别 [1, H, W]
        # squeeze(0): 移除batch维度，得到[H, W]
        prediction = output['out'].argmax(1).squeeze(0)
        
        # 将tensor转换为numpy数组，并转换为uint8类型（0-255范围）
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        
        # 将前景类别（值为1）的像素值改为255（白色），背景（值为0）保持为0（黑色）
        prediction[prediction == 1] = 255
        
        # 将numpy数组转换为PIL Image
        mask = Image.fromarray(prediction)
        
        # 保存原图
        original_path = os.path.join(output_dir, "original.jpg")
        original_img.save(original_path)
        
        # 保存分割mask
        mask_path = os.path.join(output_dir, "mask.jpg")
        mask.save(mask_path)
        
        # 创建叠加图：原图 + 彩色mask叠加
        # 将mask转换为RGB格式，并创建红色叠加层
        mask_rgb = mask.convert('RGB')
        mask_array = np.array(mask_rgb)
        original_array = np.array(original_img)
        
        # 创建红色叠加（息肉区域用红色高亮显示）
        overlay = original_array.copy()
        # 在mask为255（白色）的位置，叠加红色（透明度0.4）
        mask_bool = mask_array[:, :, 0] > 128  # 检测mask中的白色区域
        overlay[mask_bool] = (overlay[mask_bool] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
        
        # 保存叠加图
        overlay_img = Image.fromarray(overlay)
        overlay_path = os.path.join(output_dir, "overlay.jpg")
        overlay_img.save(overlay_path)
        
        print(f"原图已保存到: {original_path}")
        print(f"分割mask已保存到: {mask_path}")
        print(f"叠加图已保存到: {overlay_path}")
        
        return {
            "original": original_path,
            "mask": mask_path,
            "overlay": overlay_path,
            "inference_time": t_end - t_start
        }


if __name__ == '__main__':
    main("./data/training/images/1.jpg")
