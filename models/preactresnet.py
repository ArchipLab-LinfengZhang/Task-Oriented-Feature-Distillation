"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class DeepwiseAuxiliaryClassifier(nn.Module):
    def __init__(self, channel, num_classes=100, downsample=0):
        super(DeepwiseAuxiliaryClassifier, self).__init__()
        self.channel = channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.downsample = downsample
        self.layer = self._make_conv_layer()
        self.fc = nn.Linear(self.channel, num_classes)


    def _make_conv_layer(self):
        layer_list = []
        for i in range(self.downsample):
            layer_list.append(SepConv(self.channel, self.channel*2))
            self.channel *= 2
        layer_list.append(nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x

class PreActBasic(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, class_num=100):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deepwise1 = DeepwiseAuxiliaryClassifier(channel=64 * block.expansion, downsample=3)
        self.deepwise2 = DeepwiseAuxiliaryClassifier(channel=128 * block.expansion, downsample=2)
        self.deepwise3 = DeepwiseAuxiliaryClassifier(channel=256 * block.expansion, downsample=1)
        self.deepwise4 = DeepwiseAuxiliaryClassifier(channel=512 * block.expansion, downsample=0)
        self.stage1 = self._make_layers(block, num_block[0], 64,  1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.pre(x)
        x = self.stage1(x)
        feature_list.append(x)
        x = self.stage2(x)
        feature_list.append(x)
        x = self.stage3(x)
        feature_list.append(x)
        x = self.stage4(x)
        feature_list.append(x)

        x1 = self.deepwise1(feature_list[-4])
        x2 = self.deepwise2(feature_list[-3])
        x3 = self.deepwise3(feature_list[-2])
        x4 = self.deepwise4(feature_list[-1])

        feature = [x4, x3, x2, x1]
        x1 = self.deepwise1.fc(x1)
        x2 = self.deepwise2.fc(x2)
        x3 = self.deepwise3.fc(x3)
        x4 = self.deepwise4.fc(x4)
        return [x4, x3, x2, x1], feature


def preactresnet18():
    return PreActResNet(PreActBasic, [2, 2, 2, 2])
    
def preactresnet34():
    return PreActResNet(PreActBasic, [3, 4, 6, 3])

def preactresnet50():
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3])

def preactresnet101():
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])

def preactresnet152():
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])

