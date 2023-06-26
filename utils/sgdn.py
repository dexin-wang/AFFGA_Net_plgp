# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：test-sgdn.py
@ IDE ：PyCharm
@ Function : sgdn测试类
"""

import cv2
import os
import torch
import time
from skimage.feature import peak_local_max
import numpy as np
from utils import evaluation
from skimage.filters import gaussian
from utils.common import post_process_output
from models.loss import get_pred


RADIO = 612.0

def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


def input_depth(img, input_size=400):
    """
    对图像进行修补、裁剪，保留中间的图像块
    :param img: 深度图像, np.ndarray (h, w)
    :return: 直接输入网络的tensor, 裁剪区域的左上角坐标
    """
    assert img.shape[0] >= input_size and img.shape[1] >= input_size, '输入的深度图必须大于等于{}*{}'.format(input_size, input_size)

    img = img / 1000.0

    # 修补
    img = inpaint(img)

    # 裁剪中间的图像块
    crop_x1 = int((img.shape[1] - input_size) / 2)
    crop_y1 = int((img.shape[0] - input_size) / 2)
    crop_x2 = crop_x1 + input_size
    crop_y2 = crop_y1 + input_size
    im_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # print(im_crop.min(), im_crop.max())
    # cv2.imwrite('/home/wangdx/research/grasp_correction/sgdn/img/output/im_crop.png', depth2Gray(im_crop))

    # 归一化
    im_crop = np.clip((im_crop - im_crop.mean()), -1, 1)

    # 调整顺序，和网络输入一致
    im_crop = im_crop[np.newaxis, np.newaxis, :, :]     # (1, 1, h, w)
    im_tensor = torch.from_numpy(im_crop.astype(np.float32))  # np转tensor

    return im_tensor, crop_x1, crop_y1


def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs




class SGDN:
    def __init__(self, model, device):
        self.t = 0
        self.num = 0
        # 加载模型
        print('>> loading SGDN')
        self.device = device
        self.net = torch.load(model, map_location=torch.device(device))
        print('>> load done')

    def fps(self):
        return 1.0 / (self.t / self.num)

    def predict(self, img, device, mode, thresh=0.5, peak_dist=3, angle_k=18):
        """
        预测抓取模型
        :param img: 输入深度图 np.array (h, w)
        :param thresh: 置信度阈值
        :param peak_dist: 置信度筛选峰值
        :param angle_k: 抓取角分类数
        :return:
            pred_grasps: list([row, col, angle, width])  width单位为米
            crop_x1
            crop_y1
        """
        # 预测
        im_tensor, self.crop_x1, self.crop_y1 = input_depth(img)

        t1 = time.time()

        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net, im_tensor.to(device))

            t2 = time.time() - t1
            able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out)

            print('max graspable = ', np.max(able_pred))
            if mode == 'peak':
                # 置信度峰值 抓取点
                pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
            elif mode == 'all':
                # 超过阈值的所有抓取点
                pred_pts = arg_thresh(able_pred, thresh=thresh)
            elif mode == 'max':
                # 置信度最大的点
                loc = np.argmax(able_pred)
                row = loc // able_pred.shape[0]
                col = loc % able_pred.shape[0]
                pred_pts = np.array([[row, col]])
            else:
                raise ValueError

            # 绘制预测的抓取三角形
            pred_grasps = []
            for idx in range(pred_pts.shape[0]):
                row, col = pred_pts[idx]
                angle = angle_pred[row, col] / angle_k * np.pi  # 预测的抓取角弧度
                width = width_pred[row, col] * RADIO * 5    # 米->像素
                row += self.crop_y1
                col += self.crop_x1

                pred_grasps.append([row, col, angle, width])

        # t2 = time.time() - t1
        self.t += t2
        self.num += 1

        return pred_grasps, self.crop_x1, self.crop_y1