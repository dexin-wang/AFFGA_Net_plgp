import cv2
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
from torch.jit import Error
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/wangdx/research/sim_grasp/sgdn//')
from utils.dataset.grasp import GRASP_MAX_W



def length(pt1, pt2):
    """
    计算两点间的欧氏距离
    :param pt1: [row, col]
    :param pt2: [row, col]
    :return:
    """
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)


def diff_angle_bin(pred_angle_bin, label_angle_bins, thresh_bin=1):
    """
    判断预测的抓取角类别与抓取角标签之差是否小于阈值

    :param pred_angle_bin: 预测的抓取角类别 
    :param label_angle_bins: 一维数组 array (k, )  标注的抓取角标签概率

    :return: 
        pred_success: 预测的抓取角类别与抓取角标签之差是否小于等于阈值  
        labels: 满足抓取角类别差值小于等于阈值的抓取角类别标签
    """
    label_bins = np.argwhere(label_angle_bins == 1) # shape=(n,1)
    label_bins = np.reshape(label_bins, newshape=(label_bins.shape[0],))
    label_bins = list(label_bins)

    pred_success = False
    labels = []
    angle_k = label_angle_bins.shape[0] # 17

    for label_bin in label_bins:
        if abs(label_bin - pred_angle_bin) <= thresh_bin\
            or abs(label_bin - pred_angle_bin) >= (angle_k-thresh_bin):
            pred_success = True
            labels.append(label_bin)

    return pred_success, labels


def diff(k, label):
    """
    计算cls与label的差值
    :param k: int 不大于label的长度
    :param label: 一维数组 array (k, )  label为多标签的标注类别
    :return: min_diff: 最小的差值 int    clss_list: 角度GT的类别 len=1/2/angle_k
    """
    clss = np.argwhere(label == 1)
    clss = np.reshape(clss, newshape=(clss.shape[0],))
    clss_list = list(clss)
    min_diff = label.shape[0] + 1

    for cls in clss_list:
        min_diff = min(min_diff, abs(cls - k))

    return min_diff, clss_list


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


def rect_loc(row, col, angle, height, bottom):
    """
    计算矩形的四个角的坐标[row, col]
    :param row:矩形中点 row
    :param col:矩形中点 col
    :param angle: 抓取角 弧度
    :param height: 抓取宽度
    :param bottom: (三角形的底)
    :param angle_k: 抓取角分类数
    :return:
    """
    xo = np.cos(angle)
    yo = np.sin(angle)

    y1 = row + height / 2 * yo
    x1 = col - height / 2 * xo
    y2 = row - height / 2 * yo
    x2 = col + height / 2 * xo

    return np.array(
        [
         [y1 - bottom/2 * xo, x1 - bottom/2 * yo],
         [y2 - bottom/2 * xo, x2 - bottom/2 * yo],
         [y2 + bottom/2 * xo, x2 + bottom/2 * yo],
         [y1 + bottom/2 * xo, x1 + bottom/2 * yo],
         ]
    ).astype(np.int)


def polygon_iou(polygon_1, polygon_2):
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection / union


def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi


def evaluation_line(pred_success, pred_angle_bins, pred_width, 
                   target_success_ori, target_angle_bins_ori, target_width_ori, 
                   angle_k):
    """
    评估预测结果
    :param pred_success:    预测抓取置信度     (h, w)
    :param pred_angle_bins: 预测抓取角         (h, w)
    :param pred_width:      预测抓取宽度       (h, w)

    :param target_success:      抓取点标签      (1, 1, h, w)
    :param target_angle_bins:   抓取角标签      (1, bin, h, w)
    :param target_width:        抓取宽度标签    (1, bin, h, w)

    :param angle_k: 抓取角分类数
    :return:
        0或1, 0-预测错误, 1-预测正确

        与任意一个label同时满足以下两个条件，认为预测正确：
        1、抓取点距离小于2像素
        2、偏转角小于等于10° (angle_bin相差不大于1)
        3、抓取宽度之比在[0.8, 1.2]之内
    """
    # 阈值
    thresh_success = 0.3
    thresh_pt = 3
    thresh_angle_bin = 1
    thresh_width = 0.006

    # label预处理
    target_success   = target_success_ori[0, 0, :, :].cpu().numpy()         # (h, w)
    target_angle_bins = target_angle_bins_ori[0, :, :, :].cpu().numpy()     # (angle_k, h, w)
    target_width  = target_width_ori[0, :, :, :].cpu().numpy() * GRASP_MAX_W    # (angle_k, h, w)   m    

    # 当标签都是负样本
    # 预测成功率都是负样本时，判为正确
    # 预测成功率有正样本时，判为错误
    if np.max(target_success) < 1:
        return 1

    # 当最高置信度小于阈值时，判为检测失败
    if np.max(pred_success) < thresh_success:
        return 0

    # 获取最高置信度的抓取点
    loc = np.argmax(pred_success)
    pred_pt_row = loc // pred_success.shape[0]          # 预测的抓取点
    pred_pt_col = loc % pred_success.shape[0]           # 预测的抓取点
    pred_angle_bin = pred_angle_bins[pred_pt_row, pred_pt_col]  # 预测的抓取角类别  int
    pred_width = pred_width[pred_pt_row, pred_pt_col]      # 预测的抓取宽度 单位 m

    # 以p为中点，th为半径做圆，搜索圆内的label
    H, W = pred_success.shape
    search_l = max(pred_pt_col - thresh_pt, 0)
    search_r = min(pred_pt_col + thresh_pt, W-1)
    search_t = max(pred_pt_row - thresh_pt, 0)
    search_b = min(pred_pt_row + thresh_pt, H-1)

    for target_row in range(search_t, search_b+1):
        for target_col in range(search_l, search_r+1):
            if target_success[target_row, target_col] != 1.0:
                continue

            # 抓取点筛选
            if length([target_row, target_col], [pred_pt_row, pred_pt_col]) > thresh_pt:
                continue
            
            # 抓取角和抓取宽度
            label_angle_bins = target_angle_bins[:, target_row, target_col]   # 抓取角标签 (angle_k,)
            label_bins = np.argwhere(label_angle_bins == 1.0) # shape=(n,1)
            label_bins = np.reshape(label_bins, newshape=(label_bins.shape[0],))
            label_bins = list(label_bins)

            for label_bin in label_bins:

                if abs(label_bin - pred_angle_bin) > thresh_angle_bin\
                    and abs(label_bin - pred_angle_bin) < (angle_k-thresh_angle_bin):
                    continue
                
                target_w = target_width[label_bin, target_row, target_col]  # 抓取宽度标签
                if abs(pred_width - target_w) <= thresh_width:
                    return 1
    return 0


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


def evaluation_line_tp(pred_success_map, pred_angle_bins_map, pred_width_map, 
                   target_success_map, target_angle_bins_map, target_width_map, 
                   angle_k):
    """
    评估预测结果
    :param pred_success:    预测抓取置信度     (h, w)
    :param pred_angle_bins: 预测抓取角         (h, w)
    :param pred_width:      预测抓取宽度       (h, w)

    :param target_success:      抓取点标签      (1, 1, h, w)
    :param target_angle_bins:   抓取角标签      (1, bin, h, w)
    :param target_width:        抓取宽度标签    (1, bin, h, w)

    :param angle_k: 抓取角分类数
    :return:
        0或1, 0-预测错误, 1-预测正确

        与任意一个label同时满足以下两个条件，认为预测正确：
        1、抓取点距离小于2像素
        2、偏转角小于等于10° (angle_bin相差不大于1)
        3、抓取宽度之比在[0.8, 1.2]之内
    """
    # 阈值
    thresh_success = 0.5
    thresh_pt = 3
    thresh_angle_bin = 1
    thresh_width = 0.01

    # label预处理
    target_success_map   = target_success_map[0, 0, :, :].cpu().numpy()         # (h, w)
    target_angle_bins_map = target_angle_bins_map[0, :, :, :].cpu().numpy()     # (angle_k, h, w)
    target_width_map  = target_width_map[0, :, :, :].cpu().numpy() * GRASP_MAX_W    # (angle_k, h, w)   m    

    # 预测成功率都是负样本时，判为正确
    if np.max(target_success_map) < 1:
        return 1

    # 当最高置信度小于阈值时，判为检测失败
    if np.max(pred_success_map) < thresh_success:
        return 0

    tp = 0
    # =========== 获取预测抓取 ===========
    # all:    
    # pred_pts = arg_thresh(pred_success, thresh=thresh_success)
    # peak:
    pred_pts = peak_local_max(pred_success_map, min_distance=1, threshold_abs=thresh_success)

    # =========== 评估抓取 ===========
    pred_pts_num = pred_pts.shape[0]
    # print('pred pts num =', pred_pts_num)
    for idx in range(pred_pts_num):
        pred_pt_row, pred_pt_col = pred_pts[idx]
        pred_angle_bin = pred_angle_bins_map[pred_pt_row, pred_pt_col]  # 预测的抓取角类别  int
        pred_width = pred_width_map[pred_pt_row, pred_pt_col]      # 预测的抓取宽度 单位 m

        H, W = pred_success_map.shape
        search_l = max(pred_pt_col - thresh_pt, 0)
        search_r = min(pred_pt_col + thresh_pt, W-1)
        search_t = max(pred_pt_row - thresh_pt, 0)
        search_b = min(pred_pt_row + thresh_pt, H-1)

        success = False
        for target_row in range(search_t, search_b+1):
            for target_col in range(search_l, search_r+1):
                if target_success_map[target_row, target_col] != 1.0:
                    continue

                # 抓取点筛选
                if length([target_row, target_col], [pred_pt_row, pred_pt_col]) > thresh_pt:
                    continue
                
                # 抓取角和抓取宽度
                label_angle_bins = target_angle_bins_map[:, target_row, target_col]   # 抓取角标签 (angle_k,)
                label_bins = np.argwhere(label_angle_bins == 1.0) # shape=(n,1)
                label_bins = np.reshape(label_bins, newshape=(label_bins.shape[0],))
                label_bins = list(label_bins)

                for label_bin in label_bins:
                    if abs(label_bin - pred_angle_bin) > thresh_angle_bin\
                        and abs(label_bin - pred_angle_bin) < (angle_k-thresh_angle_bin):
                        continue
                    
                    target_w = target_width_map[label_bin, target_row, target_col]  # 抓取宽度标签
                    if abs(pred_width - target_w) <= thresh_width:
                        tp += 1.0 / pred_pts_num
                        success = True
                        break
                
                if success:
                    break
            if success:
                break
        if success:
            break
        
    return tp



def evaluation_lines(pred_success, pred_angle_bins, pred_width, 
                   target_success_ori, target_angle_bins_ori, target_width_ori, angle_k):
    """
    评估预测结果，同时记录当抓取点正确时的抓取宽度预测值

    :param pred_success:    预测抓取置信度     (h, w)
    :param pred_angle_bins: 预测抓取角         (h, w)
    :param pred_width:      预测抓取宽度       (h, w)

    :param target_success:      抓取点标签      (1, 1, h, w)
    :param target_angle_bins:   抓取角标签      (1, bin, h, w)
    :param target_width:        抓取宽度标签    (1, bin, h, w)

    :param angle_k: 抓取角分类数
    :return:

        0-三个条件均不满足
        1-满足条件1
        2-满足条件1 2
        3-满足条件1 3
        4-满足条件1 2 3
        
        与任意一个label同时满足以下两个条件，认为预测正确：
        1、抓取点距离小于2像素
        2、偏转角小于等于10° (angle_bin相差不大于1)
        3、抓取宽度之比在[0.8, 1.2]之内

        预测的抓取宽度总体偏小
    """
    # 阈值
    thresh_success = 0.3
    thresh_pt = 3
    thresh_angle_bin = 1
    thresh_width = 0.01

    target_pred_width = [[0, 0]]

    # label预处理
    target_success   = target_success_ori[0, 0, :, :].cpu().numpy()         # (h, w)
    target_angle_bins = target_angle_bins_ori[0, :, :, :].cpu().numpy()     # (angle_k, h, w)
    target_width  = target_width_ori[0, :, :, :].cpu().numpy() * GRASP_MAX_W    # (angle_k, h, w)   m    

    # 当标签都是负样本
    # 预测成功率都是负样本时，判为正确
    # 预测成功率有正样本时，判为错误
    if np.max(target_success) < 1.0:
        return 4, np.array(target_pred_width)

    # 当最高置信度小于阈值时，判为检测失败
    if np.max(pred_success) < thresh_success:
        return 0, np.array(target_pred_width)

    # 获取最高置信度的抓取点
    loc = np.argmax(pred_success)
    pred_pt_row = loc // pred_success.shape[0]          # 预测的抓取点
    pred_pt_col = loc % pred_success.shape[0]           # 预测的抓取点
    pred_angle_bin = pred_angle_bins[pred_pt_row, pred_pt_col]  # 预测的抓取角类别  int
    pred_width = pred_width[pred_pt_row, pred_pt_col]      # 预测的抓取宽度 单位 m

    H, W = pred_success.shape
    search_l = max(pred_pt_col - thresh_pt, 0)
    search_r = min(pred_pt_col + thresh_pt, W-1)
    search_t = max(pred_pt_row - thresh_pt, 0)
    search_b = min(pred_pt_row + thresh_pt, H-1)

    result = [0, 0, 0]
    result_final = [result[0], result[1], result[2]]
    max_sum = 0

    

    for target_row in range(search_t, search_b+1):
        for target_col in range(search_l, search_r+1):
        
            if target_success[target_row, target_col] != 1.0:
                continue

            # 1 抓取点筛选
            if length([target_row, target_col], [pred_pt_row, pred_pt_col]) > thresh_pt:
                continue
            result[0] = 1

            # 抓取角和抓取宽度
            label_angle_bins = target_angle_bins[:, target_row, target_col]   # 抓取角标签 (angle_k,)
            label_bins = np.argwhere(label_angle_bins == 1.0) # shape=(n,1)
            label_bins = np.reshape(label_bins, newshape=(label_bins.shape[0],))
            label_bins = list(label_bins)

            for label_bin in label_bins:
                # 抓取角
                if abs(label_bin - pred_angle_bin) <= thresh_angle_bin\
                    or abs(label_bin - pred_angle_bin) >= (angle_k-thresh_angle_bin):
                    result[1] = 1
                # 抓取宽度
                target_w = target_width[label_bin, target_row, target_col]  # 抓取宽度标签
                if abs(pred_width - target_w) <= thresh_width:
                    result[2] = 1
                
                # 记录当抓取点正确时的抓取宽度预测值
                target_pred_width.append([target_w, pred_width])
                
                if sum(result) > max_sum:
                    max_sum = sum(result)
                    result_final = [result[0], result[1], result[2]]
                result = [1, 0, 0]
            
            result = [0, 0, 0]        
            if sum(result_final) == 3:
                break
                
        if sum(result_final) == 3:
            break
    
    # 根据result_final确定返回值
    '''
    0-三个条件均不满足
    1-满足条件1
    2-满足条件1 2
    3-满足条件1 3
    4-满足条件1 2 3
    '''
    ret = 0
    if sum(result_final) == 0:
        ret = 0
    elif result_final[0] == 1 and result_final[1] == 0 and result_final[2] == 0:
        ret = 1
    elif result_final[0] == 1 and result_final[1] == 1 and result_final[2] == 0:
        ret = 2
    elif result_final[0] == 1 and result_final[1] == 0 and result_final[2] == 1:
        ret = 3
    elif sum(result_final) == 3:
        ret = 4
    else:
        print('result_final = ', result_final)
        raise Error

    return ret, np.array(target_pred_width)
