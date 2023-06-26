# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/31 18:58
@Auth ： 王德鑫
@File ：loss.py
@IDE ：PyCharm
@Function: Loss
"""
import math
import time
import torch
import torch.nn.functional as F


def mse_loss(net, x, y_pts, y_ang, y_wid):
    """
    计算二值交叉熵损失
    params:
        net: 网络
        x:     网络输入图像   (batch, 1,   h, w)
        y_pts: 抓取点标签图   (batch, 1,   h, w)
        y_ang: 抓取角标签图   (batch, bin, h, w)
        y_wid: 抓取宽度标签图 (batch, bin, h, w)
    """

    # 获取网络预测
    able_pred, angle_pred, width_pred = net(x)         # shape 同上

    # 置信度损失
    able_pred = torch.sigmoid(able_pred)
    able_loss = F.mse_loss(able_pred, y_pts)

    # 抓取角损失
    angle_pred = torch.sigmoid(angle_pred)
    angle_loss = F.mse_loss(angle_pred, y_ang)

    # 抓取宽度损失
    width_pred = torch.sigmoid(width_pred)
    width_loss = F.mse_loss(width_pred, y_wid)

    return {
        'loss': able_loss + angle_loss + width_loss,
        'losses': {
            'able_loss': able_loss,
            'angle_loss': angle_loss,
            'width_loss': width_loss,
        },
        'pred': {
            'able': able_pred,  
            'angle': angle_pred, 
            'width': width_pred,   
        }
    }


def bce_loss(net, x, y_pts, y_ang, y_wid):
    """
    计算二值交叉熵损失
    params:
        net: 网络
        x:     网络输入图像   (batch, 1,   h, w)
        y_pts: 抓取点标签图   (batch, 1,   h, w)
        y_ang: 抓取角标签图   (batch, bin, h, w)
        y_wid: 抓取宽度标签图 (batch, bin, h, w)
    """

    # 获取网络预测
    able_pred, angle_pred, width_pred = net(x)         # shape 同上

    # 置信度损失
    able_pred = torch.sigmoid(able_pred)
    able_loss = F.binary_cross_entropy(able_pred, y_pts)

    # 抓取角损失
    angle_pred = torch.sigmoid(angle_pred)
    angle_loss = F.binary_cross_entropy(angle_pred, y_ang)

    # 抓取宽度损失
    width_pred = torch.sigmoid(width_pred)
    width_loss = F.binary_cross_entropy(width_pred, y_wid)

    return {
        'loss': able_loss + angle_loss + width_loss,
        'losses': {
            'able_loss': able_loss,
            'angle_loss': angle_loss,
            'width_loss': width_loss,
        },
        'pred': {
            'able': able_pred,  
            'angle': angle_pred, 
            'width': width_pred,   
        }
    }


def bin_focal_loss(pred, target, gamma=2, alpha=0.6, width=False):
    """
    基于二值交叉熵的focal loss    
    增加难分类样本的损失
    增加负样本
    
    pred:   (N, C, H, W)
    target: (N, C, H, W)
    width: 是否输入的是抓取宽度

    对于抓取点和抓取角，置信度越大，alpha越大。置信度=0，alpha=0.4; 置信度=1，alpha=0.6。 
    对于抓取宽度，y=0时，alpha=0.4，其他alpha都等于0.6
    """
    n, c, h, w = pred.size()

    _loss = -1 * target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred+1e-7)    # (N, C, H, W)
    _gamma = torch.abs(pred - target) ** gamma

    # 根据target 设置 alpha, 形状与pred相同，
    if not width:
        k = 2.0 * alpha - 1.0   # 0.2
        b = alpha - k   # 0.4
        _alpha = k * target + b     # 0.4 - 0.6
    else:
        zeros_loc = torch.where(target == 0)
        _alpha = torch.ones_like(pred) * alpha
        _alpha[zeros_loc] = 1 - alpha

    loss = _loss * _gamma * _alpha
    loss = loss.sum() / (n*c*h*w)
    return loss


def focal_loss(net, x, y_pts, y_ang, y_wid):
    """
    计算 focal loss
    params:
        net: 网络
        x:     网络输入图像   (batch, 1,   h, w)
        y_pts: 抓取点标签图   (batch, 1,   h, w)
        y_ang: 抓取角标签图   (batch, bin, h, w)
        y_wid: 抓取宽度标签图 (batch, bin, h, w)
    """

    # 获取网络预测
    able_pred, angle_pred, width_pred = net(x)         # shape 同上

    # 置信度损失
    able_pred = torch.sigmoid(able_pred)
    able_loss = bin_focal_loss(able_pred, y_pts, width=True, alpha=0.99) * 100

    # 抓取角损失
    angle_pred = torch.sigmoid(angle_pred)
    angle_loss = bin_focal_loss(angle_pred, y_ang, width=True, alpha=0.99) * 1000

    # 抓取宽度损失
    width_pred = torch.sigmoid(width_pred)
    width_loss = bin_focal_loss(width_pred, y_wid, width=True, alpha=0.99) * 2000

    return {
        'loss': able_loss + angle_loss + width_loss,
        'losses': {
            'able_loss': able_loss,
            'angle_loss': angle_loss,
            'width_loss': width_loss,
        },
        'pred': {
            'able': able_pred,  
            'angle': angle_pred, 
            'width': width_pred,   
        }
    }



def get_pred(net, xc):
    able_pred, angle_pred, width_pred = net(xc)
    
    able_pred = torch.sigmoid(able_pred)
    angle_pred = torch.sigmoid(angle_pred)
    width_pred = torch.sigmoid(width_pred)
    return able_pred, angle_pred, width_pred
