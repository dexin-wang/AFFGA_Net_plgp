import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, upSize, angle_cls):
        """
        :param num_classes:
        :param backbone:
        :param BatchNorm:
        :param upSize: 320
        """
        super(Decoder, self).__init__()

        self.upSize = upSize
        self.angleLabel = angle_cls

        # feat_1 卷积
        self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        # 抓取置信度预测
        self.able_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(128, 1, kernel_size=1, stride=1))

        # 角度预测
        self.angle_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(128, self.angleLabel, kernel_size=1, stride=1))
        
        # 抓取宽度预测
        self.width_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(128, self.angleLabel, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, x, feat_1):
        """
        :param x: ASPP的输出特征
        :param feat_1:
        :param feat_2:
        :return:
        """
        # feat_1 卷积
        feat_1 = self.conv1(feat_1) # 96
        feat_1 = self.bn1(feat_1)
        feat_1 = self.relu(feat_1)

        # x 与 feat_1 融合
        x = F.interpolate(x, size=feat_1.size()[2:], mode='bilinear', align_corners=True)  # 上采样4倍（双线性插值）
        x = torch.cat((x, feat_1), dim=1)   # 融合 256 + 48 = 304

        able_pred = self.able_conv(x)
        angle_pred = self.angle_conv(x)
        width_pred = self.width_conv(x)

        return able_pred, angle_pred, width_pred

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm, upSize, angle_cls):
    return Decoder(num_classes, backbone, BatchNorm, upSize, angle_cls)
