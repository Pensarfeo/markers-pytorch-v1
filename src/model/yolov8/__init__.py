import math
import torch
import torch.nn as nn

from .blocks import Conv, C2f, SPPF

# class BackBone(nn.Module):
#   def __init__(self):
#     super().__init__()


#   def forward(self, x):
#     print(x)


class Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.c0, cn = convLayer(3, 64 // 4)
    self.cb2, cn = convBottlenecLayer(cn, 128//4, 1)
    self.cb4, cn4 = convBottlenecLayer(cn, 256//4, 2)
    self.cb6, cn6 = convBottlenecLayer(cn4, 512//4, 2)
    self.cb8, cn8 = convBottlenecLayer(cn6, 512//4, 1)

    self.sppf = SPPF(cn8, cn8)

    self.u12 = NeckSectionUp(cn8, cn6, 512//4)
    self.u15 = NeckSectionUp(self.u12.cn, cn4, 256//4)

    self.d18 = NeckSectionConv(self.u15.cn, self.u12.cn, 512//4)
    self.d21 = NeckSectionConv(self.d18.cn, cn8, 1024//4)
    self.h22 = DetectBlock(self.u15.cn, self.u15.cn)
    self.h23 = DetectBlock(self.d18.cn, self.u15.cn)
    self.h24 = DetectBlock(self.d21.cn, self.u15.cn)
    # self.d2 = NeckSectionConv(512, 1024, 1024)


  def forward(self, x):
    l0 = self.c0(x) # 0
    l2 = self.cb2(l0) # 5
    l4 = self.cb4(l2) # 6
    l6 = self.cb6(l4) # 7
    l8 = self.cb8(l6) # 8
    l9 = self.sppf(l8)

    l12 = self.u12(l9, l6)
    l15 = self.u15(l12, l4)
    l18 = self.d18(l15, l12)
    l21 = self.d21(l18, l9)
    l22 = self.h22(l15)
    l23 = self.h23(l18)
    l24 = self.h24(l21)
    
    return l22, l23, l24

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.01)
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        # torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      # elif isinstance(m, nn.Linear):
      #     m.weight.data.normal_(0, 0.01)
      #     m.bias.data.zero_()

