import math
import torch
import torch.nn as nn

from .blocks import ConvLayer, ConvBottlenecLayer, NeckSectionUp, SPPF, upconv
# class BackBone(nn.Module):
#   def __init__(self):
#     super().__init__()


#   def forward(self, x):
#     print(x)

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.c0 = ConvLayer(3, 64 // 4)
    self.cb2 = ConvBottlenecLayer(self.c0, 128//4, 1)
    self.cb4 = ConvBottlenecLayer(self.cb2, 256//4, 2)
    self.cb6 = ConvBottlenecLayer(self.cb4, 512//4, 2)
    self.cb8 = ConvBottlenecLayer(self.cb6, 512//4, 1)

    self.sppf = SPPF(self.cb8.cn, self.cb8.cn)

    self.u12 = NeckSectionUp(self.cb8, self.cb6, 512//4)
    self.u15 = NeckSectionUp(self.u12, self.cb4, 256//4)
    self.u18 = NeckSectionUp(self.u15, self.cb2, 128//4)

    self.u21 = NeckSectionUp(self.u18, self.c0, 64//4)
    self.u22 = upconv(self.u21, 1)
    self.output = nn.Sigmoid()

    self._initialize_weights()

  def forward(self, x):
    l0 = self.c0(x) # 0
    l2 = self.cb2(l0) # 5
    l4 = self.cb4(l2) # 6
    l6 = self.cb6(l4) # 7
    l8 = self.cb8(l6) # 8
    l9 = self.sppf(l8)

    l12 = self.u12(l9, l6)
    l15 = self.u15(l12, l4)
    l18 = self.u18(l15, l2)
    l21 = self.u21(l18, l0)
    l22 = self.u22(l21)
    # output = self.output(l22)
    return l22
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # m.weight.data.normal_(0, 0.01)
        # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

      # elif isinstance(m, nn.Linear):
      #     m.weight.data.normal_(0, 0.01)
      #     m.bias.data.zero_()

