# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import cv2
import mmcv
import math
import numpy as np
import scipy.io as scio
import scipy.stats as ss
import skimage.transform as skt
from random import choice
import random


def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

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


class DepthImage:
    def __init__(self, file):
        """
        file: mat文件
        """
        self.img = scio.loadmat(file)['A'].astype(np.float32) / 1000.      # (480, 640)  float max=0.6
        self.camera_height = 0.6

    def height(self):
        return self.img.shape[0]

    def width(self):
        return self.img.shape[1]

    def crop(self, bbox):
        """
        裁剪
        args:
            bbox: list(x1, y1, x2, y2)
        """
        self.img = self.img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def rescale_2d(self, scale, interpolation='bilinear'):
        """
        二维 resize, 只改变深度图像尺寸，不改变深度值。
        模拟不同的物体尺寸

        scale: 缩放尺度
        """
        self.img = mmcv.imrescale(self.img, scale, interpolation=interpolation)
    
    # def rescale_3d(self, height, interpolation='bilinear'):
    #     """
    #     三维 resize, 同时修改深度图像尺寸和深度值。
    #     模拟相机离桌面的不同高度，当高度变化时，抓取标签不变

    #     height: 相机离桌面的高度
    #     """
    #     resize_h = 480. * height / self.camera_height
    #     scale = 480. / resize_h
    #     self.img = mmcv.imrescale(self.img, scale, interpolation=interpolation)

    #     self.camera_height = height

    #     return scale

    def rotate(self, rota, border_value=0.6):
        """
        顺时针旋转 rota (角度)
        """
        self.img = mmcv.imrotate(self.img, rota, border_value=border_value)

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        self.img = mmcv.imflip(self.img, direction=flip_direction)

    def gaussian_noise(self, im_depth):
        """
        在image上添加高斯噪声，参考dex-net代码

        im_depth: 浮点型深度图，单位为米
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = choice([0.002, 0.003, 0.004])   # 0.004
        gaussian_process_scaling_factor = 8.0

        im_height, im_width = im_depth.shape
        
        # 1
        # mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1) # 生成一个接近1的随机数，shape=(1,)
        # mult_samples = mult_samples[:, np.newaxis]
        # im_depth = im_depth * np.tile(mult_samples, [im_height, im_width])  # 把mult_samples复制扩展为和camera_depth同尺寸，然后相乘
        
        # 2
        gp_rescale_factor = gaussian_process_scaling_factor     # 4.0
        gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
        gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0
        gp_num_pix = gp_sample_height * gp_sample_width     # im_height * im_width / 16.0
        gp_sigma = gaussian_process_sigma

        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)  # 生成(均值为0，方差为scale)的gp_num_pix个数，并reshape
        # print('高斯噪声最大误差:', gp_noise.max())
        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize成图像尺寸，bicubic为双三次插值算法
        # gp_noise[gp_noise < 0] = 0
        # camera_depth[camera_depth > 0] += gp_noise[camera_depth > 0]
        im_depth += gp_noise

        return im_depth

    def add_missing_val(self, im_depth):
        """
        添加缺失值
        梯度大的位置，概率大

        im_depth: 浮点型深度图，单位为米
        """
        global grad
        # 获取梯度图
        ksize = 11   # 调整ksize 11
        grad_X = np.abs(cv2.Sobel(im_depth, -1, 1, 0, ksize=ksize))    
        grad_Y = np.abs(cv2.Sobel(im_depth, -1, 0, 1, ksize=ksize))
        grad = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)
        
        # cv2.imshow('gradient', tool.depth2Gray(grad))
        # cv2.imwrite('D:/research/grasp_detection/sim_grasp/realsense/grad11.png', tool.depth2Gray(grad))

        # 产生缺失值的概率与梯度值呈正比
        im_height, im_width = im_depth.shape
        gp_rescale_factor = 8.0     # 调整
        gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
        gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0

        # 计算梯度对应的概率，线性方程
        """
        Sobel ksize 与 g1 g2 的对应关系
        ksize = 11  --> 300.0  3000.0
        ksize = 7  --> 1  30
        """
        min_p = 0.001
        max_p = 1.0
        g1, p1 = 500.0, min_p
        g2, p2 = 1500.0, max_p
        prob_k = (p2 - p1) / (g2 - g1)
        prob_b = p2 - prob_k * g2

        prob = grad * prob_k + prob_b
        prob = np.clip(prob, 0, max_p)

        # 生成0-1随机数矩阵
        random_mat = np.random.rand(gp_sample_height, gp_sample_width)
        random_mat = imresize(random_mat, gp_rescale_factor, interp="bilinear")  # 放大
        # cv2.imshow('random_mat', tool.depth2Gray(random_mat))

        # random_mat小于prob的位置缺失
        im_depth[ random_mat < prob ] = 0.0

        return im_depth

    def noise_gau(self):
        """
        添加高斯噪声
        """
        self.img = self.gaussian_noise(self.img)    # 添加高斯噪声

    def noise_miss(self):
        """
        添加缺失值
        """
        self.img = self.add_missing_val(self.img)   # 添加缺失值
        self.img = inpaint(self.img, missing_value=0) # 补全

    def fuzz(self):
        """
        模糊
        连续两次resize
        """
        for i in range(20):
            self.img = imresize(self.img, 0.5, interp='bilinear')
            self.img = imresize(self.img, 2.0, interp='bilinear')
        
    def slope(self):
        """
        给输入图像加入斜坡函数

        img: np.array shape=(H, W)
        
        即生成一个shape与img相同，但值为倾斜桌面深度值的二维数组
        二维数组值的方程:z = a1 * (x - W/2) + a2 * (y - H/2) + c
        """
        H, W = self.img.shape
        X = np.arange(0, W)
        Y = np.arange(0, H)
        X, Y = np.meshgrid(X, Y)

        a1 = random.random() * 0.0001
        a2 = random.random() * 0.0001

        Z = a1 * (X - W/2 + 0.5) + a2 * (Y - H/2 + 0.5)
        # print(Z.shape)
        self.img = self.img + Z

    def nomalise(self):
        self.img = np.clip((self.img - self.img.mean()), -1.0, 1.0)


class RGBImage:
    def __init__(self, file):
        """
        file: png 文件
        """
        self.img = cv2.imread(file)    # (480, 640, 3)

    def height(self):
        return self.img.shape[0]

    def width(self):
        return self.img.shape[1]

    def crop(self, bbox):
        """
        裁剪
        args:
            bbox: list(x1, y1, x2, y2)
        """
        self.img = self.img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

    def rescale(self, scale, interpolation='bilinear'):
        """
        scale: 缩放尺度
        """
        self.img = mmcv.imrescale(self.img, scale, interpolation=interpolation)
    
    def rotate(self, rota, border_value=0):
        """
        顺时针旋转 rota (角度)
        """
        self.img = mmcv.imrotate(self.img, rota, border_value=border_value)

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        self.img = mmcv.imflip(self.img, direction=flip_direction)

    def _Hue(self, img, bHue, gHue, rHue):
        # 1.计算三通道灰度平均值
        imgB = img[:, :, 0]
        imgG = img[:, :, 1]
        imgR = img[:, :, 2]

        # 下述3行代码控制白平衡或者冷暖色调，下例中增加了b的分量，会生成冷色调的图像，
        # 如要实现白平衡，则把两个+10都去掉；如要生成暖色调，则增加r的分量即可。
        bAve = cv2.mean(imgB)[0] + bHue
        gAve = cv2.mean(imgG)[0] + gHue
        rAve = cv2.mean(imgR)[0] + rHue
        aveGray = (int)(bAve + gAve + rAve) / 3

        # 2计算每个通道的增益系数
        bCoef = aveGray / bAve
        gCoef = aveGray / gAve
        rCoef = aveGray / rAve

        # 3使用增益系数
        imgB = np.expand_dims(np.floor((imgB * bCoef)), axis=2)
        imgG = np.expand_dims(np.floor((imgG * gCoef)), axis=2)
        imgR = np.expand_dims(np.floor((imgR * rCoef)), axis=2)

        dst = np.concatenate((imgB, imgG, imgR), axis=2)
        dst = np.clip(dst, 0, 255).astype(np.uint8)

        return dst

    def color(self, hue=10):
        """
        色调hue、亮度 增强
        """

        # 调节色调
        hue = np.random.uniform(-1 * hue, hue)

        if hue == 0:
            # 一般的概率保持原样 / 白平衡
            if np.random.rand() < 0.5:
                # 白平衡
                self.img = self._Hue(self.img, hue, hue, hue)
        else:
            # 冷暖色调
            bHue = hue if hue > 0 else 0
            gHue = abs(hue)
            rHue = -1 * hue if hue < 0 else 0
            self.img = self._Hue(self.img, bHue, gHue, rHue)


        # 调节亮度
        bright = np.random.uniform(-40, 10)
        imgZero = np.zeros(self.img.shape, self.img.dtype)
        self.img = cv2.addWeighted(self.img, 1, imgZero, 2, bright)

    def nomalise(self):
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()



class Image:
    def __init__(self, depthimg:DepthImage=None, rgbimg:RGBImage=None):
        self.depthimg = depthimg
        self.rgbimg = rgbimg
    
    def width(self):
        if self.rgbimg is not None:
            return self.rgbimg.width()
        if self.depthimg is not None:
            return self.depthimg.width()
    
    def height(self):
        if self.rgbimg is not None:
            return self.rgbimg.height()
        if self.depthimg is not None:
            return self.depthimg.height()
    
    def crop(self, size, dist_x=-1, dist_y=-1):
        """
        裁剪

        args:
            size: int
            dist_x: int
            dist_y: int
        return:
            crop_x1, ...
        """
        # 计算裁剪的范围
        if dist_x > 0 and dist_y > 0:
            x_offset = np.random.randint(-1 * dist_x, dist_x)
            y_offset = np.random.randint(-1 * dist_y, dist_y)
        else:
            x_offset = 0
            y_offset = 0
        crop_x1 = int((self.width() - size) / 2 + x_offset)
        crop_y1 = int((self.height() - size) / 2 + y_offset)
        crop_x2 = crop_x1 + size
        crop_y2 = crop_y1 + size

        if self.depthimg is not None:
            self.depthimg.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        if self.rgbimg is not None:
            self.rgbimg.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        return crop_x1, crop_y1, crop_x2, crop_y2

    def rescale(self, scale, interpolation='bilinear'):
        """
        scale: 缩放尺度
        """
        if self.depthimg is not None:
            self.depthimg.rescale_2d(scale, interpolation)
        if self.rgbimg is not None:
            self.rgbimg.rescale(scale, interpolation)

    def rotate(self, rota):
        """
        逆时针旋转 rota (角度)
        """
        rota = -1 * rota
        if self.depthimg is not None:
            self.depthimg.rotate(rota)
        if self.rgbimg is not None:
            self.rgbimg.rotate(rota)

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        if self.depthimg is not None:
            self.depthimg.flip(flip_direction)
        if self.rgbimg is not None:
            self.rgbimg.flip(flip_direction)

    def noise_gau(self):
        """
        添加高斯噪声和缺失值
        """
        if self.depthimg is not None:
            self.depthimg.noise_gau()
    
    def noise_miss(self):
        """
        添加高斯噪声和缺失值
        """
        if self.depthimg is not None:
            self.depthimg.noise_miss()

    def noise_fuzz(self):
        """
        模糊噪声
        """
        if self.depthimg is not None:
            self.depthimg.fuzz()
    
    def noise_slope(self):
        """
        倾斜噪声
        """
        if self.depthimg is not None:
            self.depthimg.slope()

    def color(self):
        """
        色调hue、亮度 增强
        """
        if self.rgbimg is not None:
            self.rgbimg.color()

    def nomalise(self):
        if self.depthimg is not None:
            self.depthimg.nomalise()
        if self.rgbimg is not None:
            self.rgbimg.nomalise()
            # 调整RGB图像的维度顺序
            img_rgb = self.rgbimg.img.transpose((2, 0, 1))  # (H, W, 3) -> (3, H, W)

        if self.depthimg is not None and self.rgbimg is not None:
            return np.concatenate((np.expand_dims(self.depthimg.img, 0), img_rgb), 0)   # (4, H, W)
        elif self.depthimg is not None:
            return self.depthimg.img    # (H, w)
        elif self.rgbimg is not None:
            return img_rgb  # (3, H, W)
        