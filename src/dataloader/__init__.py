import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import PIL
from torch.utils.data import Dataset, DataLoader
from .elipsesEq import drawElipses
import torch
import random

def getAnnotations(path):
    annotations = os.path.join(path, 'annotations', 'instancesMerged.json')
    
    with open(annotations, encoding='utf-8') as f:
        rawData = json.load(f)
    
    # few images have sizes under 256 in side, we just remove them
    data = []
    for d in rawData:
      h = d['height']
      w = d['width']
      if not (h < 256 or w < 256):
          data.append(d)

    return data

def toTorch(array):
   return torch.tensor(np.array(array))

def collate_fn(batch):
  images, masks, bboxMask, colors, bboxes = list(zip(*batch))  # transposed
  return toTorch(images), toTorch(masks), toTorch(bboxMask), toTorch(colors)

def noop(*args):
   return None

def crop256Patch(image):
  [dx, dy, _] = image.shape
  dl = 256
  x0 = random.randint(0, dx - dl)
  y0 = random.randint(0, dy - dl)
  return image[x0:x0+dl, y0:y0+dl]


   

class DatasetManager(Dataset):
  def __init__(self, dataDir, transform=None, target_transform=None):
      super().__init__()
      self.dataDir = dataDir
      self.imagesCache = {}
      
      self.imgDir = os.path.join(dataDir, 'imagesNormalized')
      
      self.transform = transform
      self.target_transform = target_transform
  
  def init(self):
    self.images = getAnnotations(self.dataDir)
    self.init = noop


  def __len__(self):
      # this is 5000 (5000/8 = 625)
      # return 8*10
      # return 3
      return len(self.images)

  def getImage(self, idx):
    imageName = self.images[idx]['file_name']
    imagePath = os.path.join(self.imgDir, imageName)
    imageData = cv2.imread(imagePath)
    imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
    imageData = crop256Patch(imageData)
    return imageData

  def __getitem__(self, idx):
    imageData = self.getImage(idx)
    return drawElipses(imageData)
      # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
      # image = read_image(img_path)
      # label = self.img_labels.iloc[idx, 1]
      # if self.transform:
      #     image = self.transform(image)
      # if self.target_transform:
      #     label = self.target_transform(label)
      # return image, label

datasetManagers = {}

class DataProvider():
  def __init__(self, dataDir):
    self.dataset = None
    if (dataDir not in datasetManagers):
      datasetManagers[dataDir] = DatasetManager(dataDir)
    self.dataset = datasetManagers[dataDir]
    
    self.dataset.init()
  
  def getDataLoader(self, bz):
    return DataLoader(
      self.dataset,
      batch_size = bz,
      shuffle=True,
      num_workers=0,
      collate_fn=collate_fn,
      drop_last=True
    )
     
     
