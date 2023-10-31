import math
import torch
import torch.nn as nn

from .blocks import ConvLayer, ConvBottlenecLayer, NeckSectionUp, NeckSectionConv, DetectBlock, SPPF, upconv
# class BackBone(nn.Module):
#   def __init__(self):
#     super().__init__()


#   def forward(self, x):
#     print(x)



class YoloV8BackBone(nn.Module):
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



  def forward(self, x):
    layers = [0 for i in range(16)]
    layers[0] = self.c0(x) # 0
    layers[2] = self.cb2(layers[0]) # 5
    layers[4] = self.cb4(layers[2]) # 6
    layers[6] = self.cb6(layers[4]) # 7
    layers[8] = self.cb8(layers[6]) # 8
    layers[9] = self.sppf(layers[8])

    layers[12] = self.u12(layers[9], layers[6])
    layers[15] = self.u15(layers[12], layers[4])


    
    # output = self.output(l22)
    
    return layers



class SemanticHead(nn.Module):
  def __init__(self, backbone):
    super().__init__()
    self.u18 = NeckSectionUp(backbone.u15, backbone.cb2, 128//4)
    self.u21 = NeckSectionUp(self.u18, backbone.c0, 64//4)
    self.u22 = upconv(self.u21, 1)
    self.output = nn.Sigmoid()

  def forward(self, bb):

    l18 = self.u18(bb[15], bb[2])
    l21 = self.u21(l18, bb[0])
    l22 = self.u22(l21)
    return l22

class DetectionNeck(nn.Module):
  def __init__(self, bb):
    super().__init__()
    self.u15 = bb.u15
    self.u18 = NeckSectionConv(bb.u15, bb.u12, 512//4)
    self.u21 = NeckSectionConv(self.u18, bb.sppf, 1024//4)

  def forward(self, bb):
    layers = bb + [0 for i in range(16, 22)]
    # import pdb; pdb.set_trace();
    # len(layers)
    # neck
    layers[18] = self.u18(layers[15], layers[12])
    layers[21] = self.u21(layers[18], layers[9])
    return layers

class DetectionHead(nn.Module):
  def __init__(self, neck):
    super().__init__()
    self.h22 = DetectBlock(neck.u15)
    self.h23 = DetectBlock(neck.u18)
    self.h24 = DetectBlock(neck.u21)
  
  def forward(self, body):
    h22 = self.h22(body[15])
    h23 = self.h23(body[18])
    h24 = self.h24(body[21])

    return [h22, h23, h24]


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = YoloV8BackBone()
    self.semanticHead = SemanticHead(self.backbone)
    self.detectionNeck = DetectionNeck(self.backbone)
    self.detectionHead = DetectionHead(self.detectionNeck)

  def forward(self, x):
    backbone = self.backbone(x)
    semanticHead = self.semanticHead(backbone)
    detectionBody = self.detectionNeck(backbone)
    detectionHead = self.detectionHead(detectionBody)
    return semanticHead, detectionHead

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

      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()













