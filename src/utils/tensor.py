import torch

def tile(tensor, sx = 256, sy = 256):
  dx, dy, _ = tensor.shape
  ntx = dx//sx
  nty = dy//sy

  tiles = []
  
  for i in range(ntx):
    for j in range(nty):
        tiles.append(tensor[i*sx:(i+1)*sx, j*sy:(j+1)*sy])
  
  return torch.stack(tiles), [ntx, nty, sx, sy]

def untile(tensor, tilingInfo):
    ntx, nty, sx, sy = tilingInfo
    c = 1
    if len(tensor.shape) == 4:
        c = tensor.shape[3]

    untiled = torch.zeros([sx*ntx, sy*nty, c])
    untiled = untiled.squeeze(2)

    for i in range(ntx):
        for j in range(nty):
            untiled[i*sx:(i+1)*sx, j*sy:(j+1)*sy] = tensor[i*nty + j]
    return untiled

def makeDownscaler(d = 4, n = 1):
  ds = torch.nn.Conv2d(n, n, d, d, bias=False, groups=n)
  weigths = torch.ones((n, n, d, d))/(d**2)
  ds.weight = torch.nn.Parameter(weigths)

  return ds

def cxcywhToXyxy(tensor, locTensor):
    cxcy = locTensor - tensor[:, :, :, :2]
    whHalf = tensor[:, :, :, 2:] * 0.5
    lt = cxcy - whHalf
    rb = cxcy + whHalf
    return lt, rb

def locTensor(bz, size):
    locTensor = torch.arange(0, size).unsqueeze(0).unsqueeze(2)
    locTensorX = locTensor.expand(bz, -1, size)
    locTensorY = torch.transpose(locTensorX, 2, 1)
    locTensorY.shape, locTensorY[0, 0, :], locTensorY[0, :, 0]
    return torch.stack([locTensorX, locTensorY], 3).to('cuda')

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

