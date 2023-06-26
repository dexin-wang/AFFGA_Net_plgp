from models.deeplabv3.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, input_channels, output_stride, BatchNorm, pretrained):

    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, input_channels, BatchNorm, pretrained)
        # return resnet.ResNet50(output_stride, input_channels, BatchNorm, pretrained)

    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
