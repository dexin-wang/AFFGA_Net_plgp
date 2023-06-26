# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import mmcv
import numpy as np
import cv2
import math
import scipy.io as scio


HEIGHT = 480
WIDTH = 640
GRASP_MAX_W = 0.1  # m


def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - round((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width]
    angle: 弧度
    width: 单位 像素
    mode: 显示模式 'line' or 'region'
    """

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        if mode == 'line':
            width = width / 2
            angle2 = calcAngle2(angle)
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx
            if angle < math.pi:
                cv2.line(img, (col, row), (round(col + dx), round(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (round(col - dx), round(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (round(col + dx), round(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (round(col - dx), round(row + dy)), (0, 0, 255), 1)

        # color_b = 255 / num * i
        # color_r = 0
        # color_g = -255 / num * i + 255

        color_b = 0
        color_r = 0
        color_g = 255

        if mode == 'line':
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
        else:
            img[row, col] = [color_b, color_g, color_r]
        

    return img

def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             flag=cv2.INTER_NEAREST,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=flag, borderValue=border_value)
    return rotated



class GraspMat:

    def __init__(self, file, angle_k=18):
        """
        file: *grasp.txt文件
        """
        self.angle_k = angle_k
        # 读取grasp.txt文件
        self.grasp_point_map = np.zeros((HEIGHT, WIDTH), np.float)
        self.grasp_angle_map = np.zeros((self.angle_k, HEIGHT, WIDTH), np.float)
        self.grasp_width_map = np.zeros((self.angle_k, HEIGHT, WIDTH), np.float)

        # 如果txt文件的最后一行没有字符，则f.readline()会返回None
        mode = 0
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                strs = line.split(' ')
                if len(strs) < 2:
                    mode += 1
                    continue
                if len(strs) == 4:
                    row, col, angle_bin, width = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3])
                if len(strs) == 5:
                    row, col, angle_bin, width, depth = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3]), float(strs[4])
                
                self.grasp_point_map[row, col] = 1.0 if mode in [0, 1] else 0.9
                self.grasp_angle_map[angle_bin, row, col] = 1.0 if mode == 0 else  0.9
                self.grasp_width_map[angle_bin, row, col] = width


    def height(self):
        return HEIGHT

    def width(self):
        return WIDTH


    def rescale_2d(self, scale, interpolation='nearest'):
        """
        二维放大
        """
        ori_shape = self.grasp_point_map.shape[0]

        self.grasp_point_map = mmcv.imrescale(self.grasp_point_map, scale, interpolation=interpolation)
        self.grasp_angle_map = np.stack([mmcv.imrescale(grasp_angle_map, scale, interpolation=interpolation) for grasp_angle_map in self.grasp_angle_map])
        self.grasp_width_map = np.stack([mmcv.imrescale(grasp_width_map, scale, interpolation=interpolation) for grasp_width_map in self.grasp_width_map])

        # 计算缩放比例
        new_shape = self.grasp_point_map.shape[0]
        ratio = new_shape / ori_shape
        # 抓取宽度同时缩放
        self.grasp_width_map = self.grasp_width_map * ratio
        self.grasp_angle_map[self.grasp_width_map > GRASP_MAX_W] = 0.0
        self.grasp_width_map[self.grasp_width_map > GRASP_MAX_W] = 0.0

        temp = np.sum(self.grasp_angle_map, axis=0)
        self.grasp_point_map[temp == 0] = 0
    
    
    # def rescale_3d(self, scale, interpolation='nearest'):
    #     """
    #     三维缩放
    #     """
    #     self.grasp_point_map = mmcv.imrescale(self.grasp_point_map, scale, interpolation=interpolation)
    #     self.grasp_angle_map = np.stack([mmcv.imrescale(grasp_angle_map, scale, interpolation=interpolation) for grasp_angle_map in self.grasp_angle_map])
    #     self.grasp_width_map = np.stack([mmcv.imrescale(grasp_width_map, scale, interpolation=interpolation) for grasp_width_map in self.grasp_width_map])


    def rotate(self, rota):
        """
        最近邻插值 逆时针旋转

        rota: 角度
        """
        # imrotate 是顺时针旋转
        # rota_r = -1 * rota / 180. * math.pi    # 随机选择一个角度进行旋转
        rota_r = -1 * rota
        self.grasp_point_map = imrotate(self.grasp_point_map, rota_r)
        self.grasp_angle_map = np.stack([imrotate(grasp_angle_map, rota_r) for grasp_angle_map in self.grasp_angle_map])
        self.grasp_width_map = np.stack([imrotate(grasp_width_map, rota_r) for grasp_width_map in self.grasp_width_map])

        # 逆时针旋转rota
        # 根据rota对grasp_angle_map进行平移
        # !!! 同时要对 grasp_width_map 和 grasp_dewpth_map 进行平移 
        offset = int(rota / (180. / self.angle_k))   # 0 1 -1 2 -2 3 -3
        if offset not in [0, -1, 1, 2, -2, 3, -3]:
            raise EnvironmentError

        # offset 为正数时，列表下移；offset为负数时，列表上移
        self.grasp_angle_map_new = np.zeros_like(self.grasp_angle_map)
        self.grasp_width_map_new = np.zeros_like(self.grasp_width_map)
        if offset != 0:  # 下移
            self.grasp_angle_map_new[:offset, :, :] = self.grasp_angle_map[-1*offset:, :, :]
            self.grasp_angle_map_new[offset:, :, :] = self.grasp_angle_map[:-1*offset, :, :]

            self.grasp_width_map_new[:offset, :, :] = self.grasp_width_map[-1*offset:, :, :]
            self.grasp_width_map_new[offset:, :, :] = self.grasp_width_map[:-1*offset, :, :]
        else:
            self.grasp_angle_map_new[:, :, :] = self.grasp_angle_map[:, :, :]
            self.grasp_width_map_new[:, :, :] = self.grasp_width_map[:, :, :]
        
        self.grasp_angle_map[:, :, :] = self.grasp_angle_map_new[:, :, :]
        self.grasp_width_map[:, :, :] = self.grasp_width_map_new[:, :, :]


    def crop(self, bbox):
        """
        裁剪 self.grasp

        args:
            bbox: list(x1, y1, x2, y2)
        """
        self.grasp_point_map = self.grasp_point_map[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        self.grasp_angle_map = self.grasp_angle_map[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        self.grasp_width_map = self.grasp_width_map[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]


    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        self.grasp_point_map = mmcv.imflip(self.grasp_point_map, direction=flip_direction)
        self.grasp_angle_map = np.stack([mmcv.imflip(grasp_angle_map, direction=flip_direction) for grasp_angle_map in self.grasp_angle_map])
        self.grasp_width_map = np.stack([mmcv.imflip(grasp_width_map, direction=flip_direction) for grasp_width_map in self.grasp_width_map])

        # 抓取角翻转，除了位置翻转，角度值也需要翻转
        self.grasp[2, :, :] = self._flipAngle(self.grasp[2, :, :], self.grasp[0, :, :])     # 没写完!!!!!!!


    # TODO: not done
    def _flipAngle(self, angle_mat, confidence_mat):
        """
        水平翻转angle

        Args:
            angle_mat: (h, w) 弧度
            confidence_mat: (h, w) 抓取置信度
        Returns:
        """
        # 全部水平翻转
        angle_out = (angle_mat // math.pi) * 2 * math.pi + math.pi - angle_mat
        # 将非抓取区域的抓取角置0
        angle_out = angle_out * confidence_mat
        # 所有角度对2π求余
        angle_out = angle_out % (2 * math.pi)

        return angle_out


    def nomalise(self):
        """
        抓取宽度归一化
        """
        self.grasp_point_map = self.grasp_point_map[np.newaxis, :, :]
        self.grasp_width_map = np.clip(self.grasp_width_map / GRASP_MAX_W, 0., 1.)
