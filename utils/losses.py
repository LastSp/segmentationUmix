import torch
import torch.nn as nn


class DiceLoss(nn.Module):
  def __init__(self, eps = 1e-9):
    super(DiceLoss, self).__init__()
    self.eps = eps

  def forward(self, inputs, targets):
    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    inter = (inputs * targets).sum()
    union = inputs.sum() + targets.sum()

    dice = (2*inter + self.eps) / (union + self.eps)

    return 1 - dice

class BCEDiceLoss(nn.Module):
  def __init__(self):
    super(BCEDiceLoss, self).__init__()
    self.bce = nn.BCEWithLogitsLoss()
    self.dice = DiceLoss()

  def forward(self, inputs, targets):
    dice_loss = self.dice(inputs, targets)
    bce_loss = self.bce(inputs, targets)

    return bce_loss + dice_loss

class IoULoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(IoULoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):

    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth)/(union + smooth)

    return 1 - IoU

class JaccardBCELoss(nn.Module):
  def __init__(self):
    super(JaccardBCELoss, self).__init__()
    self.bce = nn.BCEWithLogitsLoss()
    self.jaccard = IoULoss()

  def forward(self, inputs, targets):
    jaccard_loss = self.jaccard(inputs, targets)
    bce_loss = self.bce(inputs, targets)

    return bce_loss + jaccard_loss

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, inputs, targets, alpha=0.8, gamma=2.0, smooth=1):
    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    BCE = nn.functional.binary_cross_entropy(inputs, targets)
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha*(1-BCE_EXP)**gamma * BCE

    return focal_loss

class JaccardDiceLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.dice = DiceLoss()
    self.jaccard = IoULoss()

  def forward(self, inputs, targets):
    jaccard_loss = self.jaccard(inputs, targets)
    dice_loss = self.dice(inputs, targets)

    return dice_loss + jaccard_loss