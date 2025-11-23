"""
自定义数据集类，用于加载图像分割任务的数据

"""
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    PyTorch数据集类，用于加载图像和对应的分割标签
    
    Args:
        root (str): 数据集根目录路径
        train (bool): 是否为训练集，True表示训练集，False表示测试集
        transforms: 数据增强变换函数，可选
    """
    
    def __init__(self, root: str, train: bool, transforms=None):
        super(MyDataset, self).__init__()
        # 根据train参数确定使用训练集还是测试集
        self.flag = "training" if train else "test"
        # 构建数据目录路径：root/data/training 或 root/data/test
        data_root = os.path.join(root, "data", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        
        self.transforms = transforms
        
        # 获取所有jpg格式的图像文件名
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        
        # 构建图像文件的完整路径列表
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        
        # 构建标签文件的完整路径列表（标签文件名与图像文件名相同，但位于labels目录）
        self.manual = [os.path.join(data_root, "labels", i.split(".")[0] + ".jpg")
                       for i in img_names]
        
        # 检查所有标签文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (图像, 标签mask)，经过transforms处理后的数据
        """
        # 加载图像并转换为RGB格式
        img = Image.open(self.img_list[idx]).convert('RGB')
        
        # 加载标签图像并转换为灰度格式（L模式）
        manual = Image.open(self.manual[idx]).convert('L')
        
        # 将标签图像转换为numpy数组并归一化到0-1范围
        manual = np.array(manual) / 255.0
        
        # 确保mask值在0-1范围内（防止浮点误差）
        mask = np.clip(manual, a_min=0.0, a_max=1.0)
        
        # 将归一化后的mask转回PIL Image格式（需要先乘以255并转换为uint8类型）
        # 因为transforms中的操作需要PIL Image格式
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        # 应用数据增强变换（如果提供了transforms）
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        """
        返回数据集的大小
        
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        """
        自定义的批处理函数，用于将不同尺寸的图像和标签填充到相同尺寸
        
        Args:
            batch: 一个batch的数据，每个元素是(img, mask)元组
            
        Returns:
            tuple: (批处理后的图像tensor, 批处理后的标签tensor)
        """
        images, targets = list(zip(*batch))
        # 使用fill_value=0填充图像，fill_value=255填充标签（255通常表示背景或忽略区域）
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    """
    将不同尺寸的图像列表填充到相同尺寸并堆叠成batch
    
    Args:
        images: 图像tensor列表，每个tensor可能具有不同的尺寸
        fill_value: 填充值，默认为0
        
    Returns:
        tensor: 堆叠后的batch tensor，形状为(batch_size, channels, max_height, max_width)
    """
    # 计算所有图像中的最大高度和最大宽度
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    
    # 构建batch的形状：(batch_size, channels, max_height, max_width)
    batch_shape = (len(images),) + max_size
    
    # 创建填充后的tensor，使用fill_value填充
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    
    # 将每个图像复制到对应位置（左上角对齐）
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    
    return batched_imgs

