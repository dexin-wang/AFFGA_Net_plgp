import torch
import numpy as np
from skimage.filters import gaussian
import sys
sys.path.append('/home/wangdx/research/sim_grasp/sgdn/')
from utils.dataset.grasp import GRASP_MAX_W


def post_process_output(able_pred, angle_pred, width_pred):
    """
    :param able_pred:  (1, 1, h, w)           (as torch Tensors)
    :param angle_pred: (1, angle_k, h, w)     (as torch Tensors)
    :param width_pred: (1, angle_k, h, w)     (as torch Tensors)
    """

    # 抓取置信度
    able_pred = able_pred.squeeze().cpu().numpy()    # (h, w)
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)

    # 抓取角
    angle_pred = np.argmax(angle_pred.cpu().numpy().squeeze(), 0)   # (h, w)    每个元素表示预测的抓取角类别

    # 根据抓取角类别获取抓取宽度
    size = angle_pred.shape[0]
    cols = np.arange(size)[np.newaxis, :]
    cols = cols.repeat(size, axis=0)
    rows = cols.T
    width_pred = width_pred.squeeze().cpu().numpy() * GRASP_MAX_W  # (angle_k, h, w)  实际长度
    width_pred = width_pred[angle_pred, rows, cols] # (h, w)

    return able_pred, angle_pred, width_pred

