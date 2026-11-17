import torch.nn as nn

from uavdet3d.model.model_utils.norm_layer import build_norm
from uavdet3d.model.model_utils.mixstyle import build_mixstyle


class ResNet8x(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet8x, self).__init__()

        self.in_channels, self.out_channels, self.feature_channels = model_cfg.INPUT_CHANNELS, model_cfg.OUT_CHANNELS, model_cfg.NUM_FILTERS

        # 归一化类型：'bn'(缺省，与历史逐位一致) / 'gn' / 'in'。见 model_utils/norm_layer.py 的说明
        self.norm_type = model_cfg.get('NORM_TYPE', 'bn')

        self.init_block = nn.Conv2d(self.in_channels, self.feature_channels[0], kernel_size=1)

        self.block1 = self._make_block(self.feature_channels[0], self.feature_channels[0])
        self.d1 = self.down_block(self.feature_channels[0], self.feature_channels[1])

        self.block2 = self._make_block(self.feature_channels[1], self.feature_channels[1])
        self.d2 = self.down_block(self.feature_channels[1], self.feature_channels[2])

        self.block3 = self._make_block(self.feature_channels[2], self.feature_channels[2])
        self.d3 = self.down_block(self.feature_channels[2], self.feature_channels[3])

        self.block4 = self._make_block(self.feature_channels[3], self.feature_channels[3])

        self.final_conv = nn.Conv2d(self.feature_channels[3], self.out_channels, kernel_size=3, padding=1)

        # MixStyle：只插在浅层（stage 1/2 之后），深层混会伤判别力。不配时是 None，逐位等同于没有它
        self.mixstyle, self.mixstyle_layers = build_mixstyle(model_cfg)

    def _make_block(self, in_channels, feature_channels):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1))
            layers.append(build_norm(feature_channels, self.norm_type))
            layers.append(nn.ReLU(inplace=True))
            in_channels = feature_channels

        return nn.Sequential(*layers)

    def down_block(self, in_channels, feature_channels):
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
                             build_norm(feature_channels, self.norm_type),
                             nn.ReLU(inplace=True))

    def forward(self, batch_dict):
        x = batch_dict['image']  # B,K,3,W,H

        B, K, C, W, H = x.shape

        x = x.reshape(B * K, C, W, H)

        x = self.init_block(x)
        x1 = self.block1(x)
        x1 = x1 + x
        x1 = self.d1(x1)

        if self.mixstyle is not None and 1 in self.mixstyle_layers:
            x1 = self.mixstyle(x1)

        x2 = self.block2(x1)
        x2 = x2 + x1
        x2 = self.d2(x2)

        if self.mixstyle is not None and 2 in self.mixstyle_layers:
            x2 = self.mixstyle(x2)

        x3 = self.block3(x2)
        x3 = x3 + x2
        x3 = self.d3(x3)

        x4 = self.block4(x3)
        x4 = x4 + x3

        x = self.final_conv(x4)

        BK, C, W, H = x.shape

        x = x.reshape(B, K, C, W, H)

        batch_dict['features_2d'] = x

        return batch_dict


class ResNet50(nn.Module):
    def __init__(self, model_cfg):
        super(ResNet8x, self).__init__()

        self.in_channels, self.out_channels, self.feature_channels = model_cfg.INPUT_CHANNELS, model_cfg.OUT_CHANNELS, model_cfg.NUM_FILTERS

    def forward(self, batch_dict):
        x = batch_dict['image']  # B,K,3,W,H

        B, K, C, W, H = x.shape

        x = x.reshape(B * K, C, W, H)

        # x = resnet50(x)

        BK, C, W, H = x.shape

        x = x.reshape(B, K, C, W, H)

        batch_dict['features_2d'] = x

        return batch_dict
