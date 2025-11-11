"""
表情识别预测脚本
基于VGG16网络和小波变换特征融合的表情识别预测
支持7种表情分类：anger, disgust, fear, happy, neutral, sad, surprise
"""

import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pywt
from model import vgg
import numpy as np
import torch.nn.functional as F


def generate_input_heatmap(fused_tensor):
    """
    生成输入热力图用于可视化
    
    参数:
        fused_tensor: 融合后的输入张量
    
    返回:
        input_image: 归一化后的图像数组，用于可视化
    """
    # 将输入张量转换为numpy数组用于可视化
    input_image = fused_tensor.squeeze().cpu().numpy()

    # 如果输入图像有多个通道，转换为灰度图
    if input_image.ndim == 3:
        # 通过平均通道转换为灰度图
        input_image = np.mean(input_image, axis=0)

    # 归一化输入图像用于可视化（归一化到0-1范围）
    input_image -= input_image.min()
    input_image /= input_image.max()

    return input_image


def extract_wavelet_features(image_path, wavelet='haar', level=1):
    """
    使用小波变换提取图像特征
    
    参数:
        image_path: 图像文件路径
        wavelet: 小波基函数类型，默认为'haar'
        level: 小波分解层级，默认为1
    
    返回:
        approx_coeffs: 小波分解的近似系数（特征）
    """
    # 打开图像并转换为灰度图
    image = Image.open(image_path).convert('L')
    image1 = np.array(image)  # 转换为numpy数组
    
    # 执行多级小波分解
    coeffs = pywt.wavedec2(image1, wavelet, level=level)

    # 提取近似系数作为特征
    approx_coeffs = coeffs[0]
    
    return approx_coeffs


def main(img_path):
    """
    主预测函数：对输入图像进行表情识别
    
    参数:
        img_path: 输入图像文件路径
    
    返回:
        label_pre: 预测的表情类别名称（字符串）
        num: 预测的置信度（浮点数，0-1之间）
    
    功能:
        - 加载并预处理图像
        - 提取小波特征
        - 融合图像特征和小波特征
        - 使用VGG16模型进行预测
        - 返回预测结果和置信度
    """
    # ==================== 设备配置 ====================
    # 检测并设置计算设备（优先使用GPU，否则使用CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==================== 特征提取 ====================
    # 使用小波变换提取图像特征
    wave = extract_wavelet_features(img_path)

    # ==================== 图像预处理 ====================
    # 定义图像预处理流程（与训练时保持一致）
    data_transform = transforms.Compose([
        transforms.Resize(256),  # 调整大小为256x256
        transforms.CenterCrop(48),  # 中心裁剪为48x48
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    # ==================== 加载和预处理图像 ====================
    # 检查图像文件是否存在
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    
    # 打开图像
    img = Image.open(img_path)
    
    # 应用预处理变换 [C, H, W]
    img = data_transform(img)
    
    # 扩展批次维度 [1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # ==================== 特征融合 ====================
    # 将小波特征转换为张量
    tensor = torch.from_numpy(wave)
    
    # 创建两个张量
    # 注意：img 的形状是 [1, 3, 48, 48]，需要去掉批次维度得到 [3, 48, 48]
    tensor1 = img.squeeze(0)  # 形状为 [3, 48, 48] - 图像特征（去掉批次维度）
    tensor2 = tensor  # 形状为 [24, 24] 或 [48, 48] - 小波特征（取决于小波分解层级）
    
    # 调整 tensor2 的形状以匹配 tensor1
    # 如果小波特征尺寸小于图像尺寸，使用双线性插值进行上采样
    # 扩展维度: [H, W] -> [1, 1, H, W] -> 插值到 [1, 1, 48, 48] -> squeeze(0) 得到 [1, 48, 48]
    tensor2_resized = F.interpolate(
        tensor2.unsqueeze(0).unsqueeze(0), 
        size=(48, 48), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)  # 结果形状: [1, 48, 48]
    
    # 将两个张量融合（逐元素相加）
    # tensor1: [3, 48, 48], tensor2_resized: [1, 48, 48]
    # PyTorch会自动广播 tensor2_resized 到 [3, 48, 48] 然后相加
    fused_tensor = tensor1 + tensor2_resized  # 结果形状: [3, 48, 48]
    
    # 重新添加批次维度: [3, 48, 48] -> [1, 3, 48, 48]
    fused_tensor = fused_tensor.unsqueeze(0)

    # ==================== 类别映射 ====================
    # 定义类别索引到类别名称的映射（7种表情）
    class_indict = {
        "0": "angry",      # 愤怒
        "1": "disgust",    # 厌恶
        "2": "fear",       # 恐惧
        "3": "happy",      # 快乐
        "4": "neutral",    # 中性
        "5": "sad",        # 悲伤
        "6": "surprise",   # 惊讶
    }

    # ==================== 模型加载 ====================
    # 创建VGG16模型，指定7个输出类别
    model = vgg(model_name="vgg16", num_classes=7).to(device)

    # 加载训练好的模型权重
    weights_path = r"D:\code\Project\Computer_Vision\Classification\vggnet\ModelFile\vgg.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # ==================== 模型预测 ====================
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 预测时不需要计算梯度
        # 前向传播，获取模型输出
        output = torch.squeeze(model(fused_tensor.to(torch.float32).to(device))).cpu()
        
        # 应用softmax获取概率分布
        predict = torch.softmax(output, dim=0)
        
        # 获取最大概率值（置信度）
        num = round(torch.max(predict).numpy().tolist(), 4)

        # 获取预测类别索引
        predict_cla = torch.argmax(predict).numpy()
        
        # 可选：生成并保存输入热力图（已注释）
        # input_heatmap = generate_input_heatmap(fused_tensor)
        # plt.imshow(input_heatmap, cmap='jet', alpha=0.5)
        # plt.colorbar()
        # plt.title("Input Layer Heatmap")
        # plt.savefig('heatmap.png')
        # plt.close()

    # ==================== 返回结果 ====================
    # 根据预测类别索引获取类别名称
    label_pre = class_indict[str(predict_cla)]
    
    # 返回预测的类别名称和置信度
    return label_pre, num


# ==================== 主函数 ====================
if __name__ == '__main__':
    # 测试预测函数（使用示例图像）
    test_image_path = r'D:\code\Project\Computer_Vision\Classification\vggnet\datasets\data\angry\angry0001.jpg'
    result = main(test_image_path)
    print(f"预测结果: {result[0]}, 置信度: {result[1]}")
