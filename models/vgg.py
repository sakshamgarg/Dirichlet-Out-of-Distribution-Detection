'''VGG11/13/16/19 in Pytorch.'''
# modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch.nn as nn
from torch.autograd import Variable
from bbn import VarConv2d, VarLinear
import torch
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class VarVGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VarVGG, self).__init__()
        self.classifier = self._make_layers(cfg[vgg_name], True)
    
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers += [Flatten(),
            nn.Dropout(),
            VarLinear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            VarLinear(512, 512),
            nn.ReLU(True),
            VarLinear(512, 10)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.classifier(x)
        return out, None

    def get_KL(self):
        KL = torch.tensor(0.0).cuda()
        for layer in self.classifier:
            if "Var" in type(layer).__name__:
                KL += layer.get_KL()
        return KL

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        #self.classifier = nn.Linear(512, num_classes)
        self.confidence = nn.Linear(512, 1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        pred = self.classifier(out)
        confidence = self.confidence(out)
        return pred, confidence

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
