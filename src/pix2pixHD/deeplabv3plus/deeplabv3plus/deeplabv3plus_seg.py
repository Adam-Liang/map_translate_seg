"""
deeplabv3+ only for Segmentation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pix2pixHD.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.backbone import build_backbone
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.ASPP import ASPP


class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # for l in layers:
        #     print(l.shape)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        # print(feature_aspp.shape)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature_map = self.cat_conv(feature_cat)
        result = self.cls_conv(feature_map)
        result = self.upsample4(result)
        return result,feature_map

    def get_paras(self):
        backbone_params=self.backbone.parameters()
        base_params = list(map(id, self.backbone.parameters())) # 注意此处不能使用backbone_params，会导致该迭代器被疯狂使用，后面无法通过正确性检验
        global_params = filter(lambda p: id(p) not in base_params, self.parameters())
        # num_bb=sum(1 for _ in backbone_params) # 分割正确性检验
        # num_gl=sum(1 for _ in global_params)
        # num_all=sum(1 for _ in self.parameters())
        return global_params,backbone_params


def get_params(model, key):
    print('????')
    for m in model.named_modules():
        print(m)
        if key == '1x':
            if (any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                      nn.Conv2d):
                print(m[0])
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if (not any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                          nn.Conv2d):
                for p in m[1].parameters():
                    yield p


if __name__ == '__main__':
    pass
    from models.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plus(cfg)
    # get_params(model=model, key='1x')
    key = '10x'
    for m in model.named_modules():
        if key == '1x':
            if (any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                      nn.Conv2d):
                print(m[0])
                for p in m[1].parameters():
                    pass
        elif key == '10x':
            if (not any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                          nn.Conv2d):
                print(m[0])
                for p in m[1].parameters():
                    pass