from torchvision.ops import DropBlock2d
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
  def __init__(self, in_planes, ratio=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.fc(self.avg_pool(x))
    max_out = self.fc(self.max_pool(x))
    out = avg_out + max_out
    return self.sigmoid(out)

class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()

    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x_2 = torch.cat([avg_out, max_out], dim=1)
    x_2 = self.conv1(x_2)
    return self.sigmoid(x_2) * x

class solo_conv(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.dropblock = DropBlock2d(p=0.1, block_size=7)
    self.bn1 = nn.BatchNorm2d(out_c)
    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.dropblock(x)
    x = self.bn1(x)
    x = self.relu(x)

    return x

class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_c)

    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_c)

    self.dropblock = DropBlock2d(p=0.1, block_size=7)
    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.dropblock(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.dropblock(x)
    x = self.bn2(x)
    x = self.relu(x)

    return x

class encoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.conv = conv_block(in_c, out_c)
    self.pool = nn.MaxPool2d((2, 2))

  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)

    return x, p

class decoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    self.conv = conv_block(out_c+out_c, out_c)

  def forward(self, inputs, skip):
    x = self.up(inputs)
    #print(x.shape, skip.shape)
    x = torch.cat([x, skip], axis=1)
    x = self.conv(x)
    return x

class SA_UNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.e1 = encoder_block(3, 16)
    self.e2 = encoder_block(16, 32)
    self.e3 = encoder_block(32, 64)
    #self.e4 = encoder_block(64, 128)

    self.b1 = solo_conv(64, 128)
    self.att = SpatialAttention()
    self.b2 = solo_conv(128, 128)

    self.d1 = decoder_block(128, 64)
    self.d2 = decoder_block(64, 32)
    self.d3 = decoder_block(32, 16)

    self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)

  def forward(self, inputs):
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)

    b1 = self.b1(p3)
    a = self.att(b1)
    b2 = self.b2(a)

    d1 = self.d1(b2, s3)
    d2 = self.d2(d1, s2)
    d3 = self.d3(d2, s1)

    outputs = self.outputs(d3)

    return outputs

class SA_UNet_2(nn.Module):
  def __init__(self):
    super().__init__()

    self.e1 = encoder_block(3, 16)
    self.e2 = encoder_block(16, 32)
    self.e3 = encoder_block(32, 64)
    self.e4 = encoder_block(64, 128)

    self.b1 = solo_conv(128, 256)
    self.att = SpatialAttention()
    self.b2 = solo_conv(256, 256)

    self.d1 = decoder_block(256, 128)
    self.d2 = decoder_block(128, 64)
    self.d3 = decoder_block(64, 32)
    self.d4 = decoder_block(32, 16)

    self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)

  def forward(self, inputs):
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)

    b1 = self.b1(p4)
    a = self.att(b1)
    b2 = self.b2(a)

    d1 = self.d1(b2, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)

    outputs = self.outputs(d4)

    return outputs