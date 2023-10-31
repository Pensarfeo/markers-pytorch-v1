import os

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import addSrcToPath
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

# from model.yolov8 import Model
from model.yolov8.semantic_detection_d0 import Model
from dataloader import DataProvider
# import the time module
import time

batch_size = 8

writer = SummaryWriter('runs/netVisualize')
model = Model()
writer.add_graph(model, torch.zeros((1, 3, 512, 512)))

# print('model:', model)
# # torch.save(model.state_dict(), './save.pt')
pred = model(torch.zeros((1, 3, 512, 512)))
# print(model)

