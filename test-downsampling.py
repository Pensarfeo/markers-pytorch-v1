import addSrcToPath
import numpy as np
import os
import random
import torch
import time
from dataloader.elipsesEq import drawElipses
import cv2
from tqdm import tqdm
import math
random.seed(time.time())

purple =  [158, 68, 98]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

purpleTensor = torch.Tensor(purple).to(device, dtype=torch.float)
purpleTensor = purpleTensor.unsqueeze(0).unsqueeze(0).expand(640, 640, 3)

imageOriginal = cv2.imread('test1.jpg')

def getDraw():

  image = np.copy(imageOriginal)
  xmin = random.randint(0, image.shape[0] - 256)
  ymin = random.randint(0, image.shape[1] - 256)
  crop256 = imageOriginal[xmin:xmin+256, ymin:ymin+256]

  return drawElipses(crop256)

image1, mask1, bbox1, c, bboxes = getDraw()
image2, mask2, bbox2, c, bboxes = getDraw()
image3, mask3, bbox3, c, bboxes = getDraw()

mask = np.stack([mask1, mask2, mask3], axis=0)
image = np.stack([image1, image2, image3], axis=0)
c = np.stack([c, c, c], axis=0)


norm255 = 1/255

def makeDownscaler(d = 4, n = 1):
  ds = torch.nn.Conv2d(n, n, d, d, bias=False, groups=n)
  weigths = torch.ones((n, n, d, d))/(d**2)
  ds.weight = torch.nn.Parameter(weigths)

  return ds



with torch.no_grad():
  _, dx, dy = mask.shape
  x = torch.from_numpy(image).to(device, dtype=torch.float)
  y = torch.from_numpy(mask).to(device, dtype=torch.float).unsqueeze(3)
  c = torch.from_numpy(c).unsqueeze(1).unsqueeze(1)
  c = c.expand(-1, dx, dy, -1)
  colorCentered = torch.abs(x)/255
  print(y.shape, x.shape)
  output = colorCentered[0].cpu().numpy()
  print('y.shape', y.shape)
  
  # downScaler8c1 = makeDownscaler(d = 8).to(device)
  # downScaler2c1 = makeDownscaler(d = 2).to(device)
  # downScaler8c4 = makeDownscaler(d = 8, 4).to(device)
  # downScaler8c4 = makeDownscaler(d = 8, 4).to(device)

  # yd8 = downScaler4(y.permute(0, 3, 1, 2))
  # yd2 = downScaler2(yd4)
  # yd16 = downScaler2(yd8)

  print('yd4:', yd8.shape, torch.max(yd8), torch.min(yd8))
  # print('c:', y.shape)

  # cv2.imshow('input', output)
  # cv2.waitKey(0) # waits until a key is pressed
  # cv2.destroyAllWindows() # destroys the window
  yd8 = torch.ceil(yd8.permute(0, 2, 3, 1))

  for i in range(3):
    outputD4 = cv2.resize(yd8[i].cpu().numpy(), [256, 256])
    side2Side = np.concatenate((mask[i], outputD4), axis=1)
    cv2.imshow('target', side2Side)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window
  

  # cv2.imshow('target', y.permute(0, 2, 3, 1)[0].cpu().numpy())
  # cv2.waitKey(0) # waits until a key is pressed
  # cv2.destroyAllWindows() # destroys the window

    # x = xn.to(device, dtype=torch.float)
    
    # pred = model(x.permute(0, 3, 1, 2))
    # pred = torch.sigmoid(torch.squeeze(pred))
    # pred = pred.cpu().numpy()


    # cv2.imshow('input', 2*np.abs(xn.numpy()[0])/255)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window


    # cv2.imshow('prediction', pred)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image

    # pred = pred.permute(0, 2, 3, 1)
    # cv2.imshow('pred', pred[0])