import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, image, target):
        #bce = F.binary_cross_entropy_with_logits(image, target)
        criterionbce = nn.BCELoss()
        bce = criterionbce(image, target)
        smooth = 1e-5
        image = torch.sigmoid(image)
        num = target.size(0)
        image = image.view(num, -1)
        target = target.view(num, -1)
        intersection = (image * target)
        dice = (2. * intersection.sum(1) + smooth) / (image.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, image, target):
        #bce = F.binary_cross_entropy_with_logits(image, target)
        smooth = 1e-5
        image = torch.sigmoid(image)
        num = target.size(0)
        image = image.view(num, -1)
        target = target.view(num, -1)
        intersection = (image * target)
        dice = (2. * intersection.sum(1) + smooth) / (image.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


'''
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(image,target)
        return bce
'''
