import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplabv3.aspp import build_aspp
from models.deeplabv3.decoder import build_decoder
from models.deeplabv3.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, input_channels, angle_cls=18, backbone='resnet', output_stride=8, num_classes=21, sync_bn=False,
                 freeze_bn=False, size=360):
        super(DeepLab, self).__init__()
        # if backbone == 'drn':
        #     output_stride = 8

        # if sync_bn:
        #     BatchNorm = SynchronizedBatchNorm2d
        # else:   # <-
        #     BatchNorm = nn.BatchNorm2d

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, input_channels, output_stride, BatchNorm, pretrained=False)      # 主干网络
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)              # ASPP
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, size, angle_cls=angle_cls)          # 解码器
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, feat_1 = self.backbone(input)
        x = self.aspp(x)
        able_pred, angle_pred, width_pred = self.decoder(x, feat_1)
 
        # able_pred = F.interpolate(able_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        # angle_pred = F.interpolate(angle_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        # width_pred = F.interpolate(width_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）

        # able_pred = self.able_out(able_pred)
        # angle_pred = self.angle_out(angle_pred)
        # width_pred = self.width_out(width_pred)

        return able_pred, angle_pred, width_pred

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


