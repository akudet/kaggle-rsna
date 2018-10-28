import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Encoder(nn.Module):
    """
    feature encoder
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class RPN(nn.Module):

    def __init__(self, in_chs, anchors):
        super().__init__()
        self.in_channels = in_chs
        self.anchors = anchors

        self.h_chs = 256
        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, self.h_chs, 3),
            nn.ReLU(),
        )
        self.rpn_class = nn.Sequential(
            nn.Conv2d(self.h_chs, 2 * len(self.anchors), 1),
            nn.ReLU(),
        )
        self.rpn_bbox = nn.Sequential(
            nn.Conv2d(self.h_chs, 4 * len(self.anchors), 1),
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x = self.conv(x)
        rpn_class = self.rpn_class(x)
        rpn_class = rpn_class.reshape(batch_size, 2, -1)
        rpn_bbox = self.rpn_bbox(x)
        rpn_bbox = rpn_bbox.reshape(batch_size, 4, -1)
        return [rpn_class, rpn_bbox]


class RoIPolling(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, feats, rois):
        output = []
        rois_num = rois.size(1)

        for i in range(rois_num):
            roi = rois[0][i]
            x, y, w, h = roi
            output.append(F.adaptive_max_pool2d(feats[:, :, y:y + h, x:x + w], self.output_size))

        return torch.cat(output)
