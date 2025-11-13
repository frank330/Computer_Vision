"""离线车牌检测与字符/颜色识别脚本。"""
# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import time
import os
import cv2
import copy
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from car_recognition.car_rec import init_car_rec_model, get_color_and_score
import torch
import numpy as np


# 彩色标记等可视化配置
gr_bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']
object_color = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]
class_type = ['单层车牌', '双层车牌', '汽车']


def order_points(pts):
    """将四个顶点按左上、右上、右下、左下排序，便于透视变换。"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """对指定四点区域进行透视变换，裁剪出车牌小图。"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def load_model(weights, device):
    """加载 YOLO 检测模型权重。"""
    return attempt_load(weights, map_location=device)


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """将关键点坐标从网络尺寸映射回原图坐标系。"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    return coords


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model, car_rec_model):
    """对单个检测框进行字符/颜色识别，并返回结构化结果。"""
    h, w, _ = img.shape
    result_dict = {}
    x1, y1, x2, y2 = map(int, xyxy)
    rect = [x1, y1, x2, y2]

    if int(class_num) == 2:  # 类别 2 表示整车，用于估计车辆颜色
        car_roi_img = img[y1:y2, x1:x2]
        car_color, color_conf = get_color_and_score(car_rec_model, car_roi_img, device)
        result_dict.update({
            'class_type': class_type[int(class_num)],
            'rect': rect,
            'score': conf,
            'object_no': int(class_num),
            'car_color': car_color,
            'color_conf': color_conf,
        })
        return result_dict

    # 类别 0/1：单层或双层车牌
    landmarks_np = np.zeros((4, 2))
    for i in range(4):
        landmarks_np[i] = np.array([int(landmarks[2 * i]), int(landmarks[2 * i + 1])])

    class_label = int(class_num)
    roi_img = four_point_transform(img, landmarks_np)
    if class_label == 1:
        roi_img = get_split_merge(roi_img)
    plate_number, plate_color = get_plate_result(roi_img, device, plate_rec_model)
    if any(d in plate_number for d in danger):
        plate_number = '危险品'

    result_dict.update({
        'class_type': class_type[class_label],
        'rect': rect,
        'landmarks': landmarks_np.tolist(),
        'plate_no': plate_number,
        'roi_height': roi_img.shape[0],
        'plate_color': plate_color,
        'object_no': class_label,
        'score': conf,
    })
    return result_dict


def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, car_rec_model=None):
    """使用检测模型对整图进行车牌/车辆检测并返回识别结果列表。"""
    conf_thres = 0.3
    iou_thres = 0.5
    outputs = []
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img)[0]
    _ = time_synchronized() - t1

    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
            for item in det:
                xyxy = item[:4].view(-1).tolist()
                conf = item[4].cpu().numpy()
                landmarks = item[5:13].view(-1).tolist()
                class_num = item[13].cpu().numpy()
                outputs.append(
                    get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model, car_rec_model)
                )
    return outputs


def draw_result(orgimg, dict_list):
    """将识别结果绘制在原图上，并返回组合字符串。"""
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        object_no = result['object_no']
        if object_no != 2:  # 车牌目标
            x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
            padding_w = 0.05 * w
            padding_h = 0.11 * h
            rect_area[0] = max(0, int(x - padding_w))
            rect_area[1] = max(0, int(y - padding_h))
            rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
            rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

            height_area = int(result['roi_height'] / 2)
            landmarks = result['landmarks']
            result_p = result['plate_no'] + " " + result['plate_color']
            if result['object_no'] == 1:
                result_p += "双层"
            result_str += result_p + " "
            for i in range(4):
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, gr_bgr[i], -1)

            if "危险品" in result_p:
                orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], rect_area[3], (0, 255, 0), height_area)
            else:
                orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0] - height_area,
                                       rect_area[1] - height_area - 10, (0, 255, 0), height_area)

        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), object_color[object_no], 2)
    return orgimg, result_str


parser = argparse.ArgumentParser()
parser.add_argument('--detect_model', nargs='+', type=str, default='weights/detect.pt', help='检测模型权重路径')
parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='车牌字符+颜色模型权重路径')
parser.add_argument('--car_rec_model', type=str, default='weights/car_rec_color.pth', help='车辆颜色识别模型权重路径')
parser.add_argument('--image_path', type=str, default='imgs/1.jpg', help='测试图片路径')
parser.add_argument('--img_size', type=int, default=384, help='推理尺寸')
parser.add_argument('--output', type=str, default='result1', help='输出目录')
parser.add_argument('--video', type=str, default='', help='视频路径（保留参数）')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()
save_path = opt.output
if not os.path.exists(save_path):
    os.mkdir(save_path)

# 初始化模型
detect_model = load_model(opt.detect_model, device)
plate_rec_model = init_model(device, opt.rec_model)
car_rec_model = init_car_rec_model(opt.car_rec_model, device)


def detect_Recognition_plate11(img_path):
    """对输入图片执行检测与识别，返回字符串结果。"""
    img = cv_imread(img_path)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size, car_rec_model)
    return draw_result(img, dict_list)[1]


if __name__ == '__main__':
    print(detect_Recognition_plate11(r'D:\code\Project\Computer_Vision\Solutions\car\imgs\2.jpg'))