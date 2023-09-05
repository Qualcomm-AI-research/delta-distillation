# ------------------------------------------------------------------------------
# Copyright (c) 2020 xingkong
# Licensed under the MIT License.
# All rights reserved.
#
# This code is from https://github.com/ydhongHIT/DDRNet
# ------------------------------------------------------------------------------
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BN_MOMENTUM
from .common import BasicBlock
from .common import BatchNorm2d
from .common import Bottleneck
from .common import DAPPM
from .common import segmenthead


class DualResNet(nn.Module):
    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3_1 = self._make_layer(block, planes * 2, planes * 4, layers[2] // 2, stride=2)
        self.layer3_2 = self._make_layer(block, planes * 4, planes * 4, layers[2] // 2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3_1 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=BN_MOMENTUM),
        )

        self.compression3_2 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=BN_MOMENTUM),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
            BatchNorm2d(highres_planes, momentum=BN_MOMENTUM),
        )

        self.down3_1 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
        )

        self.down3_2 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 8, momentum=BN_MOMENTUM),
        )

        self.layer3_1_ = self._make_layer(block, planes * 2, highres_planes, layers[2] // 2)

        self.layer3_2_ = self._make_layer(block, highres_planes, highres_planes, layers[2] // 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, layers[3])

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3_1(self.relu(x))
        layers.append(x)
        x_ = self.layer3_1_(self.relu(layers[1]))
        x = x + self.down3_1(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3_1(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear')

        x = self.layer3_2(self.relu(x))
        layers.append(x)
        x_ = self.layer3_2_(self.relu(x_))
        x = x + self.down3_2(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3_2(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear')

        temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))
        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[4])),
            size=[height_output, width_output],
            mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear')

        x_ = self.final_layer(x + x_)

        if self.augment:
            x_extra = self.seghead_extra(temp)
            return [x_, x_extra]
        else:
            return x_

    def init_weights(self, cfg):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pretrained = os.path.join(cfg.morphine_root, cfg.checkpoint.init)
        if os.path.isfile(pretrained):
            print('=> loading pretrained model {}'.format(pretrained))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.local_rank} if cfg.local_rank != -1 else None
            pretrained_dict = torch.load(pretrained, map_location=map_location)

            pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items()}
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            warnings.warn(f'checkpoint does not exist')


class DualResNet39(DualResNet):
    def __init__(self):
        super().__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=19, planes=64, spp_planes=128, head_planes=256, augment=False
        )

