import torch
import torchvision


lossCE = torch.nn.BCEWithLogitsLoss()

def lossl1(pred, target):
  size = target.shape[1]*target.shape[2]


  negTarget = (1 - target)

  shouldBePositiveArea = torch.abs((target - pred)*target)
  shouldBeNegativeArea = torch.abs((target - pred)*negTarget)*positiveRatio
  error = torch.sum(shouldBePositiveArea + shouldBeNegativeArea)/target.shape[1]

  return error

def dice(pred, target):
  pred = torch.sigmoid(pred)
  union = torch.sum(pred*target, (1, 2))
  totarea = torch.sum(pred, (1, 2)) + torch.sum(target, (1, 2))

  dice = 1 - ((2*union)/totarea)

  return dice

def semantic(pred, target):
   ce = lossCE(torch.squeeze(pred), target)
   loss = ce

  #  l1 = lossl1(pred, target)*0
  #  print('l1:', l1.item())
  
   di = dice(pred, target)
   loss += di
  
   return loss, ce, di

def bBox(predXY, predWhOriginal, targetBBox):
    errorXY = torch.abs(predXY-targetBBox[:, :, :, :2])
    errorWH = torch.abs(predWhOriginal-targetBBox[:, :, :, 2:])
    error = torch.sum(errorXY + errorWH, dim=(1, 2, 3))

    return error


# def xwhwtoxyxy(predBBox):
#    xy = predBBox[:, :, :, 0:2]
#    hw = predBBox[:, :, :, 2]*0.5
#    x1y1 = hw - xy
#    x2y2 = hw + xy
#    return torch.cat(x1y1, x2y2, dim=3)

# def wrappingBox(predBBox, targetBBox):
#     x1y1x2y2Pred = xwhwtoxyxy(predBBox)
#     x1y1x2y2Target = xwhwtoxyxy(targetBBox)
#     joined = torch.stack([x1y1x2y2Pred, x1y1x2y2Target], dim=4)

def minAbs(tensor):
   return torch.min(torch.abs(tensor))

def maxAbs(tensor):
   return torch.max(torch.abs(tensor))

def anyNaN(tensor):
   torch.any(torch.isnan(tensor))

def cxcywhToXyxy(tensor, locTensor):
    cxcy = locTensor - tensor[:, :, :, :2]
    whHalf = tensor[:, :, :, 2:] * 0.5
    lt = cxcy - whHalf
    rb = cxcy + whHalf
    return lt, rb

def calcIou(ltOutput, rbOutput, ltTarget, rbTarget):
    # Calc Intersection

    ltInter = torch.max(ltOutput, ltTarget)
    rbInter = torch.min(rbOutput, rbTarget)

    wh = (rbInter - ltInter).clamp(min=0)  # [N,M,2]
    torch.any(wh < 0)

    areaInter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [N,M]
    
    whOutput = (rbOutput - ltOutput)
    whTarget = (rbTarget - ltTarget)
    
    areaOutput = whOutput[:, :, :, 0]*whOutput[:, :, :, 1]
    areaTarget = whTarget[:, :, :, 0]*whTarget[:, :, :, 1]

    areaUnion = areaOutput + areaTarget - areaInter

    return areaInter, areaUnion

def calcMaxBox(ltMaxBox, rbMaxBox):
    whMaxBox = ltMaxBox - rbMaxBox
    areaMaxBox = whMaxBox[:, :, 0] * whMaxBox[:, :, 1]
    return areaMaxBox

def caclDIou(ltMaxBox, rbMaxBox, ltOutput, rbOutput, ltTarget, rbTarget, eps: float = 1e-7):
    # The diagonal distance of the smallest enclosing box squared
    diagonal_distance_squared = torch.sum((ltMaxBox - rbMaxBox) ** 2, dim=3) + eps

    centerOutput = (rbOutput + ltOutput)*0.5
    centerTarget = (ltTarget + rbTarget)*0.5

    # The distance between boxes' centers squared.
    centers_distance_squared = torch.sum((centerOutput - centerTarget) ** 2, dim=3)
    
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    dLoss = (centers_distance_squared / diagonal_distance_squared)
    return dLoss

def calcAIou(output, target, iou, eps: float = 1e-7,):
    w_pred = output[:, :, :, 2]
    h_pred = output[:, :, :, 3] + 1e-7
    w_gt = target[:, :, :, 2]
    h_gt = target[:, :, :, 3] + 1e-7
    
    v = (4 / (torch.pi**2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    loss = alpha * v

    return loss
def minMax(tensor):
   tensor = torch.abs(tensor)
   return torch.min(tensor), torch.max(tensor)

def completeDistanceIou(predXY, predWH, target, locTensor, targetSemantic):
    # valid = targetSemantic > 0.5
    valid = torch.logical_and(predWH[:, :, :, 0] > 0, predWH[:, :, :, 0] > 0)
    # valid = torch.logical_and(valid, nonZeroWH)
    output = torch.concat([predXY, predWH], dim=3)
    # Boxes coordinates
    ltOutput, rbOutput = cxcywhToXyxy(output, locTensor)
    ltTarget, rbTarget = cxcywhToXyxy(target, locTensor)

    ltMaxBox = torch.min(ltOutput, ltTarget)
    rbMaxBox = torch.max(rbOutput, rbTarget)


    areaMaxBox = calcMaxBox(ltMaxBox, rbMaxBox)
    areaInter, areaUnion = calcIou(ltOutput, rbOutput, ltTarget, rbTarget)

    iou = (areaInter / (areaUnion + 1e-7))
    dIou = caclDIou(ltMaxBox, rbMaxBox, ltOutput, rbOutput, ltTarget, rbTarget)
    aIou = calcAIou(output, target, iou)

    iou = 1 - iou[valid] + dIou[valid]
    return iou


def completeIouLoss(predBBox, targetBBox):
   return torchvision.ops.complete_box_iou_loss(predBBox.view(-1, 4), targetBBox.view(-1, 4), reduction = 'sum')

def locTensor(bz, size):
    locTensor = torch.arange(0, size).unsqueeze(0).unsqueeze(2)
    locTensorX = locTensor.expand(bz, -1, size)
    locTensorY = torch.transpose(locTensorX, 2, 1)
    locTensorY.shape, locTensorY[0, 0, :], locTensorY[0, :, 0]
    return torch.stack([locTensorX, locTensorY], 3).to('cuda')

  # negativeTarget = 1 - targetSemantic
  # nPositive = torch.sum(targetSemantic, dim=(1, 2), keepdim=True)
  # nNegative = torch.sum(negativeTarget, dim=(1, 2), keepdim=True)
  # positiveNegativeRatio = negativeTarget*(nPositive/nNegative) + targetSemantic

class Detection:
  def __init__(self, bz, size = 256):
    self.lossCe = torch.nn.BCEWithLogitsLoss(reduce = False)
    self.locationTensor = torch.arange(0, size*size).view([size, size])
    self.locationTensor = locTensor(bz, size)
    self.bBoxL2 = torch.nn.MSELoss()
    self.bBoxL1 = torch.nn.L1Loss()
    self.bBoxLH = torch.nn.HuberLoss(reduction='mean', delta=1.0)

  def loss(self, predSemantic, predXY, predWH, predWhOriginal, targetSemantic, targetBBox):
    
    xywh = torch.concat([predXY, predWhOriginal], dim=3)
    # valid = targetSemantic > 0.5
    # xywhPredValid = xywh[valid]
    # xywhtargValid = targetBBox[valid]

    # bxL2 = self.bBoxL2(xywhPredValid, xywhtargValid)
    # bxL1 = self.bBoxL1(xywhPredValid, xywhtargValid)
    bxHL = self.bBoxL2(xywh, targetBBox)
    bx = bxHL
    ceNorm = 1/(256**2)
    ce = self.lossCe(torch.squeeze(predSemantic), targetSemantic)
    ce = torch.sum(ce, (1, 2))
    ce = torch.mean(ce)*ceNorm

    di = dice(predSemantic, targetSemantic)
    di = torch.mean(di)

    cIou = completeDistanceIou(predXY, predWH, targetBBox, self.locationTensor, targetSemantic)
    # cIou = completeDistanceIou(targetBBox[:, :, :, :2], targetBBox[:, :, :, 2:], targetBBox, self.locationTensor, targetSemantic)*(1-targetSemantic)
    cIou = torch.mean(cIou)

    return bx, ce, di, cIou