 # coding: utf-8
import os
import time
import numpy as np
import cv2

class Rotate:
    def __init__(self, degree,size):
        self.degree=degree
        self.size=size
    def __call__(self, img):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.degree, self.size)
        img = cv2.warpAffine(img, M, (cols, rows))
        return img

class Brightness:
    def __init__(self,brightness_factor):
        self.brightness_factor=brightness_factor
    def __call__(self, img):
        img = np.uint8(np.clip((self.brightness_factor*img), 0, 255))
        return img



def alter_(path,object):
    s = os.listdir(path)
    count = 0
    for i in s:
        document = os.path.join(path,i)
        img = cv2.imread(document)
        img = cv2.resize(img, (224,224))
        cv2.imwrite(object + '%s.jpg' % i.split('.')[0], img)
        count = count + 1


def alter_flip(path,object):
    s = os.listdir(path)
    count = 0
    for i in s:
        document = os.path.join(path,i)
        img = cv2.imread(document)
        img = cv2.resize(img, (224,224))
        # 翻转
        img = cv2.flip(img,0)
        cv2.imwrite(object + 'flip%s.jpg' % i.split('.')[0], img)
        count = count + 1


def alter_Rotate(path,object):

    s = os.listdir(path)
    count = 0
    for i in s:
        document = os.path.join(path,i)
        img = cv2.imread(document)
        img = cv2.resize(img, (224,224))
        # 旋转
        rotate = Rotate(45, 0.7)
        img = rotate(img)
        cv2.imwrite(object + 'Rotate%s.jpg' % i.split('.')[0], img)
        count = count + 1
def alter_Brightness(path,object):
    s = os.listdir(path)
    count = 0
    for i in s:
        document = os.path.join(path,i)

        img = cv2.imread(document)
        img = cv2.resize(img, (224,224))
        # 亮度
        brightness = Brightness(0.9)
        img = brightness(img)
        cv2.imwrite(object + 'Brightness%s.jpg' % i.split('.')[0], img)
        count = count + 1


def alter_gaussian(path, object):
    s = os.listdir(path)
    count = 0
    for i in s:
        document = os.path.join(path, i)

        img = cv2.imread(document)
        img = cv2.resize(img, (224,224))
         # 设置高斯噪声的参数
        mean = 0  # 平均值
        std = 25  # 标准差，决定噪声的强度
         # 生成高斯噪声
        gaussian_noise = np.random.normal(mean, std, img.shape).astype('uint8')
         # 将高斯噪声添加到图片上
        noisy_image = cv2.add(img, gaussian_noise)
        cv2.imwrite(object + 'gaussian%s.jpg' % i.split('.')[0], noisy_image)
        count = count + 1
def read_name_list():
    name_list = []
    for child_dir in os.listdir("./dataset/data/"):
        name_list.append(child_dir)
    print(name_list)
    return name_list

list = read_name_list()
for i in list:
    alter_("./dataset/data\\" + i, "./dataset/data1\\" + i + "\\")
    alter_Rotate("./dataset/data\\"+i,"./dataset/data1\\"+i+"\\")
    alter_flip("./dataset/data\\" + i,"./dataset/data1\\" + i+"\\")
    alter_Brightness("./dataset/data\\" + i, "./dataset/data1\\" + i+"\\")
    alter_gaussian("./dataset/data\\" + i, "./dataset/data1\\" + i + "\\")
