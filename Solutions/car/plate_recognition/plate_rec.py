"""车牌字符与颜色识别模型封装。"""
from plate_recognition.plateNet import myNet_ocr, myNet_ocr_color
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys


def cv_imread(path):
    """支持中文路径的 cv2 读取方法。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    """递归遍历 rootPath 下所有 jpg/png 图片路径并加入列表。"""
    fileList = os.listdir(rootPath)
    for temp in fileList:
        full_path = os.path.join(rootPath, temp)
        if os.path.isfile(full_path):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(full_path)
        else:
            allFilePath(full_path, allFIleList)


# 全局设备与字符映射表
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
color_list = ['黑色', '蓝色', '绿色', '白色', '黄色']
mean_value, std_value = (0.588, 0.193)

"""CTC 是 “Connectionist Temporal Classification”（连接时序分类）的缩写，是一种常用在序列识别任务中的解码机制。
它允许模型在输出序列时自动处理“对齐”问题，通过插入空白标记并在最终结果中去除重复和空白，从而获得连续且不重复的字符序列。
当前 `decodePlate` 函数里就是在做这个“去空白、去重复”的后处理。
在 CTC 里，模型输出的是“帧级”序列：同一个字符通常会连续预测多帧（甚至穿插空白标记 `_`），因为模型每一步并不知道整串字符已经“说完”还是还在继续。
因此，后处理时需要“去空白、去重复”：
1. 去空白：CTC 会在字符集里加一个专门的 blank 标记，网络遇到字符间隔或不确定时会输出它，最终结果里需要剔除。  
2. 去重复：同一个字符如果连续多帧都预测为同一 label，CTC 默认压缩成一次（只保留第一个），这样才能得到我们习惯的“字符顺序”。  

比如预测序列 `['渝','渝','_','A','A','1','1']`，通过 CTC 规则去掉连续重复和空白，最终会解码成 `渝A1`。这正是 `decodePlate()` 在做的事情。"""
def decodePlate(preds):
    """使用 CTC 解码去重相邻字符，返回最终字符索引序列。"""
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    return newPreds


def image_processing(img, device):
    """对车牌图像做尺寸归一化与标准化，生成模型输入 tensor。"""
    img = cv2.resize(img, (168, 48))
    img = np.reshape(img, (48, 168, 3))
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model):
    """调用模型获取车牌号与颜色，返回 (plate_str, color_str)。"""
    input = image_processing(img, device)
    preds, color_preds = model(input)
    preds = preds.argmax(dim=2)
    color_preds = color_preds.argmax(dim=-1)
    preds = preds.view(-1).detach().cpu().numpy()
    color_preds = color_preds.item()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds:
        plate += plateName[i]
    return plate, color_list[color_preds]


def init_model(device, model_path):
    """加载车牌识别+颜色模型，返回初始化后的模型实例。"""
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    # model_path = os.sep.join([sys.path[0], model_path])
    model = myNet_ocr_color(num_classes=len(plateName), export=True, cfg=cfg, color_num=len(color_list))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


