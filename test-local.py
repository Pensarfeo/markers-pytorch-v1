import addSrcToPath
import numpy as np
import os
import random
import torch
import time
from model.yolov8.semantic import Model
from dataloader import DataProvider
import cv2
from utils.tensor import tile, untile
from utils.image import tilePadding

random.seed(time.time())

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# annotationsDir = os.path.join('.', 'data', 'coco', 'val2017')
# trainDataloader = DataProvider(annotationsDir, 1)

# model = torch.load('./Yolov8_bz4_Epochs1_randxy_n-5_dl-50_center_color_epochs-9.pt')

model = torch.load('./outputs/Yolov8_bz4_Epochs1_randxy_n-5_dl-50_centerColor_d-cocoTest/epochs-0.pt')
model.eval()
print(next(model.parameters()).is_cuda, next(model.parameters()).is_cuda)

purple = [98, 68, 158]
tileDx = 256
tileDy = 256

image = cv2.imread('./test1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = tilePadding(image - purple, tileDx, tileDy)

with torch.no_grad():
#   for (xn, y) in trainDataloader:
  x = torch.from_numpy(image).to(device, dtype=torch.float)
  x = torch.abs(x)/255
  x, tilingInfo = tile(x)
  print('x:', x.shape)

  pred = model(x.permute(0, 3, 1, 2))
  pred = torch.sigmoid(torch.squeeze(pred))
  print('pred:', pred.shape)
  pred = untile(pred, tilingInfo)

  print('pred:', pred.shape)
  pred = pred.cpu().numpy()

  cv2.imshow('input', image/255)
  cv2.waitKey(0) # waits until a key is pressed
  cv2.destroyAllWindows() # destroys the window

  cv2.imshow('target', pred)
  cv2.waitKey(0) # waits until a key is pressed
  cv2.destroyAllWindows() # destroys the window

  # cv2.imshow('prediction', pred)
  # cv2.waitKey(0) # waits until a key is pressed
  # cv2.destroyAllWindows() # destroys the window showing image

      # pred = pred.permute(0, 2, 3, 1)
      # cv2.imshow('pred', pred[0])