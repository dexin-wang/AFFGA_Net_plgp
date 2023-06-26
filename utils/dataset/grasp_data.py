"""
数据集加载类
"""

import numpy as np
import cv2
import math
import random
import os
import copy
import glob
from random import choice
import torch
import torch.utils.data
import sys
sys.path.append('/home/wangdx/research/sim_grasp/sgdn/')
from utils.dataset.grasp import GraspMat, drawGrasps, GRASP_MAX_W
from utils.dataset.img import Image, RGBImage, DepthImage
from utils.tool import depth2Gray3



class SimGraspDataset(torch.utils.data.Dataset):
    def __init__(self, path, database, mode, output_size, use_rgb, use_dep, start_n=0, end_n=-1, angle_k=18, argument=False):
        """
        :param path: 数据集路径 到correctiob/regression
        :param database: 数据库 str, 'egad'/'dex'/'clutter'  多个数据库之间用空格隔开
        :param mode: 'train'/'test'
        :param output_size: int 输入网络的图像尺寸
        :param use_rgb: 是否使用RGB图像作为输入
        :param use_dep: 是否使用深度图像作为输入
        :param start_n: 采用的样本在数据集中的起始索引
        :param end_n: 采用的样本在数据集中的终止索引
        :param argument: 是否数据集增强
        """
        self.output_size = output_size
        self.angle_k = angle_k
        self.use_rgb = use_rgb
        self.use_dep = use_dep
        self.argument = argument

        assert mode in ['train', 'test', 'train test']

        databases = [database] if ' ' not in database else database.split(' ')

        # 读取数据集 深度图文件列表
        grasp_files = []
        for data_base in databases:
            modes = mode.split(' ')
            for m in modes:
                dataset_path = os.path.join(path, data_base, m)
                grasp_files.extend(glob.glob(os.path.join(dataset_path, '*_grasp.txt')))
        # grasp_files.sort()

        # 获取数据集片段
        if end_n < 0 or end_n > len(grasp_files):
            self.grasp_files = grasp_files
        else:
            self.grasp_files = grasp_files[start_n:end_n]
        if len(self.grasp_files) == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(path))


    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))


    def __getitem__(self, idx):
        # 读取img和抓取标签
        grasp_name = self.grasp_files[idx]
        label = GraspMat(grasp_name, angle_k=self.angle_k)
        image = Image()
        if self.use_rgb:
            im_rgb = RGBImage(grasp_name.replace('grasp.txt', 'rgb.png'))
            image.rgbimg = im_rgb
        if self.use_dep:
            im_dep = DepthImage(grasp_name.replace('grasp.txt', 'depth.mat'))
            image.depthimg = im_dep        

        # 数据增强
        if self.argument:
            # resize
            if random.random() < 0.5:
                scale = np.random.uniform(1.0, 1.2)
                image.rescale(scale)
                label.rescale_2d(scale)
            # rotate
            if random.random() < 0.5:
                r =  180. / self.angle_k
                rota = choice([0, r, -1*r, 2*r, -2*r, 3*r, -3*r])    # 随机选择一个角度进行旋转
                image.rotate(rota)
                label.rotate(rota)

            # 裁剪为8的倍数，不然缺失值和高斯噪声无法生成
            crop_bbox = image.crop(480)
            label.crop(crop_bbox)

            # fuzz
            if random.random() < 0.5:
                image.noise_fuzz()
            
            # gaussian
            if random.random() < 0.5:
                image.noise_gau()
            
            # miss
            if random.random() < 0.5:
                image.noise_miss()
            
            # crop
            dist_x = 30   # 横向裁剪随机数
            dist_y = 30   # 纵向裁剪随机数
            crop_bbox = image.crop(self.output_size, dist_x, dist_y)
            label.crop(crop_bbox)

            # slope
            if random.random() < 0.5:
                image.noise_slope()
                
            # color
            if random.random() < 0.5:
                image.color()
        else:
            # crop
            crop_bbox = image.crop(self.output_size)
            label.crop(crop_bbox)

        # img归一化
        input = self.numpy_to_torch(image.nomalise())    # (n, h, w)
        # 获取target
        label.nomalise()
        grasp_point, grasp_angle, grasp_width = label.grasp_point_map, label.grasp_angle_map, label.grasp_width_map # (1, h, w) (bins, h, w) (bins, h, w)
        grasp_point = self.numpy_to_torch(grasp_point)
        grasp_angle = self.numpy_to_torch(grasp_angle)
        grasp_width = self.numpy_to_torch(grasp_width)

        return input, grasp_point, grasp_angle, grasp_width


    def __len__(self):
        return len(self.grasp_files)




# main函数用于测试数据集加载情况
if __name__ == '__main__':
    angle_cls = 18
    dataset_path = '/home/wangdx/dataset/sim_grasp'
    # 加载训练集
    print('Loading Dataset...')
    train_dataset = SimGraspDataset(dataset_path, database='plgpd_test', mode='train', output_size=360, use_rgb=True, use_dep=True, start_n=0, end_n=20, argument=True)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    print('>> dataset: {}'.format(len(train_data)))

    count = 0
    max_w = 0
    for x, y_pts, y_ang, y_wid in train_data:
        count += 1
        # 解码图像
        img = x.cpu().numpy().squeeze()     # (4, h, w)
        img_dep, img_rgb = img[0], img[1:]
        img_dep = (np.array(img_dep)* 1000).astype(np.uint16)

        print('img_dep.min() = ', img_dep.min())

        img_rgb = img_rgb.astype(np.uint8).transpose((1, 2, 0))
        im_rgb = np.zeros(img_rgb.shape, dtype=np.uint8)
        im_rgb[:, :, 0] = img_rgb[:, :, 0]
        im_rgb[:, :, 1] = img_rgb[:, :, 1]
        im_rgb[:, :, 2] = img_rgb[:, :, 2]

        # 解码抓取标签
        grasp_point_map = y_pts.cpu().numpy().squeeze()    # (h, w)
        grasp_angle_map = y_ang.cpu().numpy().squeeze()  # (angle_k, h, w)
        grasp_width_map = y_wid.cpu().numpy().squeeze() * GRASP_MAX_W  # (angle_k, h, w) 

        # 绘制抓取
        grasps = []
        grasp_pts = np.where(grasp_point_map > 0)
        for i in range(grasp_pts[0].shape[0]):
            if i % 2 != 0:
                continue
            row, col = grasp_pts[0][i], grasp_pts[1][i]
            angle_bins = np.where(grasp_angle_map[:, row, col] > 0)[0]
            for angle_bin in angle_bins:
                angle = (angle_bin / angle_cls) * math.pi
                width = grasp_width_map[angle_bin, row, col] / 0.0016   # 这里直接
                grasps.append([row, col, angle, width])
        
        camera_depth = depth2Gray3(img_dep)
        # camera_depth_region = camera_depth.copy()
        camera_depth_line = camera_depth.copy()
        # im_grasp_region = drawGrasps(camera_depth_region, grasps, mode='region')
        im_dep_line = drawGrasps(camera_depth_line, grasps, mode='line')
        im_rgb_line = drawGrasps(im_rgb, grasps, mode='line')
        # cv2.imwrite('/home/wangdx/research/sim_grasp/sgdn/img/test/' + str(count) + '_region.png', im_grasp_region)
        cv2.imwrite('/home/wangdx/research/sim_grasp/sgdn/img/test/' + str(count) + '_dep.png', im_dep_line)
        cv2.imwrite('/home/wangdx/research/sim_grasp/sgdn/img/test/' + str(count) + '_rgb.png', im_rgb_line)

        print('saving ...', count)