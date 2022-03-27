import torch
from torch import nn
from torch.nn import functional as F

class ResNetBlock(nn.Module):
  def __init__(self, chn,k=3, use_resnet=False, use_swish=False):
    super().__init__()
    self.l1 = nn.Conv2d(chn, chn, k, padding=1)
    self.use_resnet = use_resnet
    if use_swish:
      self.act = nn.SiLU()
    else:
      self.act = nn.ReLU()
  def forward(self, x):
    if self.use_resnet:
      return self.act(x + self.l1(x))
    else:
      return self.act(self.l1(x))

class ConvModel_1(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2)
        self.res_1 = ResNetBlock(32, 3)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.res_2 = ResNetBlock(64, 3)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(128, out_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.res_1(x)
        x = self.act(self.conv_2(x))
        x = self.res_2(x)
        x = self.act(self.conv_3(x))
        x = self.pool(x).view(x.shape[0],-1)
        return F.log_softmax(self.out(x), dim=-1)

class ConvModel_2(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2)
        self.res_1 = ResNetBlock(32, 3)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.res_2 = ResNetBlock(64, 3)
  
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(128, out_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.25)
    def forward(self, x):
        x = self.act(self.drop(self.conv_1(x)))
        x = self.res_1(x)
        x = self.act(self.drop(self.conv_2(x)))
        x = self.res_2(x)
        x = self.act(self.drop(self.conv_3(x)))
        x = self.pool(x).view(x.shape[0],-1)
        return F.log_softmax(self.out(x), dim=-1)

class ConvModel_3(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2)
        self.res_1 = ResNetBlock(32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.res_2 = ResNetBlock(64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(128, out_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.bn1(self.res_1(x))
        x = self.act(self.conv_2(x))
        x = self.bn2(self.res_2(x))
        x = self.act(self.conv_3(x))
        x = self.pool(x).view(x.shape[0],-1)
        return F.log_softmax(self.out(x), dim=-1)

class ConvModel_4(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2)
        self.res_1 = ResNetBlock(32, 3, use_swish=True)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.res_2 = ResNetBlock(64, 3, use_swish=True)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(128, out_size)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.res_1(x)
        x = self.act(self.conv_2(x))
        x = self.res_2(x)
        x = self.act(self.conv_3(x))
        x = self.pool(x).view(x.shape[0],-1)
        return F.log_softmax(self.out(x), dim=-1)

class ConvModel_5(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2)
        self.res_1 = ResNetBlock(32, 3)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.res_2 = ResNetBlock(64, 3)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(128, out_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.res_1(x)
        x = self.act(self.conv_2(x))
        x = self.res_2(x)
        x = self.act(self.conv_3(x))
        x = self.pool(x).view(x.shape[0],-1)
        return F.log_softmax(self.out(x), dim=-1)

class ConvModel_BnResnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2)
        self.res_1 = ResNetBlock(32, 3, use_resnet=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.res_2 = ResNetBlock(64, 3, use_resnet=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(128, out_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.bn1(self.res_1(x))
        x = self.act(self.conv_2(x))
        x = self.bn2(self.res_2(x))
        x = self.act(self.conv_3(x))
        x = self.pool(x).view(x.shape[0],-1)
        return F.log_softmax(self.out(x), dim=-1)

