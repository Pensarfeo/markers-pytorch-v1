import addSrcToPath
import numpy as np
import os
import random
import torch
import time
from model.yolov8.semantic import Model
from dataloader import DataProvider
import cv2
from tqdm import tqdm
random.seed(time.time())

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

annotationsDir = os.path.join('.', 'data', 'coco', 'train2017')
# batchSize = 92*10
batchSize = 92
trainDataloader = DataProvider(annotationsDir, batchSize)


norm255 = 1/255
for (x, y, c) in tqdm(trainDataloader):
  _, dx, dy = y.shape
  x = x.to(device)
  c = c.to(device)
  c = c.unsqueeze(1).unsqueeze(1)
  c = c.expand(-1, dx, dy, -1)
  colorCentered = torch.abs(x)/255
  output = colorCentered[0].cpu().numpy()
 
  # print('c:', y.shape)

  # cv2.imshow('input', output)
  # cv2.waitKey(0) # waits until a key is pressed
  # cv2.destroyAllWindows() # destroys the window

  # cv2.imshow('target', y[0].cpu().numpy())
  # cv2.waitKey(0) # waits until a key is pressed
  # cv2.destroyAllWindows() # destroys the window

    # x = xn.to(device, dtype=torch.float)
    
    # pred = model(x.permute(0, 3, 1, 2))
    # pred = torch.sigmoid(torch.squeeze(pred))
    # pred = pred.cpu().numpy()


    # cv2.imshow('input', 2*np.abs(xn.numpy()[0])/255)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window

    # cv2.imshow('target', y.numpy()[0])
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window

    # cv2.imshow('prediction', pred)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() # destroys the window showing image

    # pred = pred.permute(0, 2, 3, 1)
    # cv2.imshow('pred', pred[0])
    
for (x, y, c) in tqdm(trainDataloader):
  _, dx, dy = y.shape