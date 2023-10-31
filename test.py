import addSrcToPath
import numpy as np
import os
import random
import torch
import time
from model.yolov8.semantic import Model
from dataloader import DataProvider
import cv2
random.seed(time.time())

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

annotationsDir = os.path.join('.', 'data', 'coco', 'train2017')
trainDataloader = DataProvider(annotationsDir, 1)

model = torch.load('./Yolov8_bz4_Epochs1_randxy_n-5_dl-50_centerColor_d-cocoTest/epochs-0.pt')
model.eval()

print(next(model.parameters()).is_cuda, next(model.parameters()).is_cuda)

norm255 = 1/255
with torch.no_grad():
  for (x, y, c) in trainDataloader:
      x = x.to(device, dtype=torch.float)
      _, dx, dy = y.shape
      y = y.to(device)
      c = c.to(device)
      c = c.unsqueeze(1).unsqueeze(1)
      c = c.expand(-1, dx, dy, -1)
      x = torch.abs(x - c)*norm255
      x = x.permute(0, 3, 1, 2)

      pred = model(x)
      pred = torch.sigmoid(torch.squeeze(pred))
      pred = pred.cpu().numpy()

      # cv2.imshow('input', x[0].cpu().numpy())
      # cv2.waitKey(0) # waits until a key is pressed
      # cv2.destroyAllWindows() # destroys the window

      cv2.imshow('target', y[0].cpu().numpy())
      cv2.waitKey(0) # waits until a key is pressed
      cv2.destroyAllWindows() # destroys the window

      cv2.imshow('prediction', pred)
      cv2.waitKey(0) # waits until a key is pressed
      cv2.destroyAllWindows() # destroys the window showing image

      # pred = pred.permute(0, 2, 3, 1)
      # cv2.imshow('pred', pred[0])