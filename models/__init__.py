def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from models.ggcnn.ggcnn import GGCNN
        return GGCNN

    # GGCNN2
    elif network_name == 'ggcnn2':
        from models.ggcnn.ggcnn2 import GGCNN2
        return GGCNN2
    # DeepLabv3+
    elif network_name == 'deeplabv3':
        from models.deeplabv3.deeplab import DeepLab
        return DeepLab
    # GRCNN
    elif network_name == 'grcnn':
        from models.grcnn.grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # UNet
    elif network_name == 'unet':
        from models.unet.unet import U_Net
        return U_Net
    # SegNet
    elif network_name == 'segnet':
        from models.segnet.segnet import SegNet
        return SegNet
    # STDC
    elif network_name == 'stdc':
        from models.STDC.stdc import BiSeNet
        return BiSeNet
    # DANet
    elif network_name == 'danet':
        from models.danet.encoding.models.sseg.danet import get_danet
        return get_danet
    
    # AFFGA
    elif network_name == 'affga':
        from models.affga.deeplab import DeepLab
        return DeepLab

    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
