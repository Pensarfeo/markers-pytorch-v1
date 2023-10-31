import os

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import addSrcToPath
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

# from model.yolov8 import Model
from model.yolov8.semantic import Model
from dataloader import DataProvider
# import the time module
import time

batch_size = 8

annotationsDir = os.path.join('.', 'data', 'coco', 'val2017')
trainDataloader = DataProvider(annotationsDir, batch_size)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# writer = SummaryWriter('runs/netVisualize')


model = Model()

size_model = 0
for param in model.parameters():
    if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
print(f"model size: {size_model} / bit | {size_model / 1048576} / MB")

model.to(device)
# print('model:', model)
# # writer.add_graph(model, torch.zeros((1, 3, 512, 512)))
# # torch.save(model.state_dict(), './save.pt')
# pred = model(torch.zeros((1, 3, 512, 512)))
# # print(model)

def lossFn(pred, target):
  size = target.shape[1]*target.shape[2]
  print(  'size:', size)
  greater = torch.gt(target, 0.5)
  
  positiveSize = torch.sum(greater, dim=(1, 2), keepdim=True)
  positiveRatio = positiveSize/size
  print('positiveRatio', positiveRatio[:, 0, 0])

  weightMask = (1 - target)*positiveSize

  zeroPred = torch.zeros(pred.shape).to(device)
  errorTest = torch.sum(torch.abs(target - zeroPred)).item()

  error = torch.sum(torch.abs(target - pred))/size
  # print('errorTestDiff:', errorTest)
  # print('errorTestDiff:', error.item())
  # print('errorTestDiff:', errorTest - error.item())

  return error


optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
iter = 0 
maxIters = 100
def backProplf (x):
  lr = (1 - iter / maxIters) * (1e-4 - 1e-6) + 1e-6
  print('lr', lr)
  return lr
before_lr = optimizer.param_groups[0]["lr"]
print('beforeb_lr:', before_lr)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-3, total_iters=620)
before_lr = optimizer.param_groups[0]["lr"]
print('before_lr:', before_lr)

# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=backProplf, verbose = True)

class MovingAverage():
  def __init__(self, n):
    self.n = n
    self.cache = [0]*n
    self.i = 0
    
  def add(self, v):
    p = self.i%(self.n)
    self.cache[p] = v
    self.result = 0
    for v in self.cache:
      self.result += v
    self.result /= self.n
    self.i += 1
    return self.result

lossMovingAverage =  MovingAverage(10)
def train(dataloader, model, lossFn, optimizer):
    print('TRAINING')
    startTraining = time.time()
    model.train()
    letLastIterEndTime = None
    for i, (x, y) in enumerate(dataloader):
      iter = i
      print(i, '===>')
      # if letLastIterEndTime is not None:
        # print('iterTime =>', i, time.time() - letLastIterEndTime)
      letLastIterEndTime = time.time()
      x, y = x.to(device, dtype=torch.float), y.to(device)
      # Compute prediction error
      inputTensor = x.permute(0, 3, 1, 2)
      pred = model(inputTensor)
      # print('max_memory_allocated:', f"{torch.cuda.max_memory_allocated() /1048576}", 'MB')
      loss = lossFn(pred, y)
      lossAverage = lossMovingAverage.add(loss.item())

      with torch.no_grad():
        
        # maxTarget = torch.max(torch.flatten(y, 1), 1).values
        # print('maxVals:', maxTarget)
        flattentedPred = torch.flatten(pred.detach(), 1 )
        maxVals = torch.max(flattentedPred, 1).values
        print('maxVals:', maxVals)
        if torch.isnan(maxVals).any():
           asd

      print('lossAverage:', lossAverage, loss.item())
      loss.backward()
      optimizer.step()
      before_lr = optimizer.param_groups[0]["lr"]
      print('before_lr:', before_lr)
      scheduler.step()
      after_lr = optimizer.param_groups[0]["lr"]
      print("Epoch %d: SGD lr %.4f -> %.4f" % (i, before_lr, after_lr))
      
      del pred, x, y, loss
    torch.save(model, './loss_l1.pt')
    print('timeTraining =>', time.time() - startTraining)

    #     # Backpropagation
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     if batch % 100 == 0:
    #         loss, current = loss.item(), (batch + 1) * len(X)
    #         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# print(f"Using {device} device")

# # def test(dataloader, model, loss_fn):
# #     size = len(dataloader.dataset)
# #     num_batches = len(dataloader)
# #     model.eval()
# #     test_loss, correct = 0, 0
# #     with torch.no_grad():
# #         for X, y in dataloader:
# #             X, y = X.to(device), y.to(device)
# #             pred = model(X)
# #             test_loss += loss_fn(pred, y).item()
# #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
# #     test_loss /= num_batches
# #     correct /= size
# #     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainDataloader, model, lossFn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")