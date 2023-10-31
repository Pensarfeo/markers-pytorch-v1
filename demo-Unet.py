import os
import cv2
import addSrcToPath
from dataSaver import DataSaver
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler, RMSprop
import torch.nn.functional as F

# from model.yolov8 import Model
from model.yolov8.semantic_detection_d0 import Model as YoloV8
from model.unet import UNet
from dataloader import DataProvider
from tqdm import tqdm
from utils.dice_score import dice_loss
from utils import initialize_weights
from utils.math import MovingAverage
from utils.tensor import makeDownscaler

# import the time module
import time

modelName = os.path.join('outputs', "Yolov8-cocoTest-noise-brighness-colortwicking-counterExamples-d0")
dataSaver = DataSaver(os.path.join(modelName, 'output')) 
# batchSize = 92
batchSize = 2

learningRate = 1e-3
weightDecay = 1e-8
momentum = 0.999
amp = True
epochStart = 0
epochs = 10
accumulate = 1

def getCheckpointName(modelName, epoch):
  return os.path.join(modelName, f'epochs-{epoch}.pt')



annotationsDir = os.path.join('.', 'data', 'coco', 'train2017')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# writer = SummaryWriter('runs/netVisualize')


# model = UNet(n_channels= 3, n_classes=1)

model = None
checkpoint = None

try:
  model = torch.load(getCheckpointName(modelName,   epochStart - 1))
  print('loaded model from checkpoint')
except:
  model = YoloV8()
  initialize_weights(model)

size_model = 0
for param in model.parameters():
    if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
print(f"model size: {size_model} / bit | {size_model*1.25e-7} / MB")

model.to(device)
# print('model:', model)
# # writer.add_graph(model, torch.zeros((1, 3, 512, 512)))
# # torch.save(model.state_dict(), './save.pt')
pred = model(torch.zeros((1, 3, 512, 512)).to(device))
print('pred[0].shape', pred[0].shape)
print('pred[0].shape', pred[1].shape)


optimizer = RMSprop(
  model.parameters(),
  lr=learningRate,
  weight_decay=weightDecay,
  momentum=momentum,
  foreach=True
)

totIters = 1276
print('epochs*(totIters//batchSize//accumulate):', epochs*(totIters//batchSize//accumulate))
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.7, end_factor=1e-2, total_iters=epochs*(totIters//batchSize//accumulate))
before_lr = optimizer.param_groups[0]["lr"]
lossCE = nn.BCEWithLogitsLoss()

# L1 loss where we normalize the background to have less weight
def lossl1(pred, target):
  size = target.shape[1]*target.shape[2]
  greater = torch.gt(target, 0.5)
  
  positiveSize = torch.sum(greater, dim=(1, 2), keepdim=True)
  positiveRatio = positiveSize/size
  print('positiveRatio:', positiveRatio.squeeze((1, 2)))

  negTarget = (1 - target)

  shouldBePositiveArea = torch.abs((target - pred)*target)
  shouldBeNegativeArea = torch.abs((target - pred)*negTarget)*positiveRatio
  error = torch.sum(shouldBePositiveArea + shouldBeNegativeArea)/target.shape[1]

  return error

def lossDice(pred, target):
  pred = torch.sigmoid(pred)
  union = torch.sum(pred*target, (1, 2))
  totarea = torch.sum(pred, (1, 2)) + torch.sum(target, (1, 2))
  dice = torch.sum(1 - ((2*union)/totarea))
  return dice

def lossClass(pred, target):
   loss = torch.zeros([1]).to(device)
   
   ce = lossCE(torch.squeeze(pred), torch.squeeze(target, dim=(1, 2), keepdim=True))
   loss += ce

  #  l1 = lossl1(pred, target)*0
  #  print('l1:', l1.item())
  
   di = lossDice(pred, target)
   loss += di
  
   return loss, ce, di
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)


def lossBbox(pred, targetBbox, targetClass):
  import pdb; pdb.set_trace()
  return torch.abs(targetBbox-pred)/torch.sum(targetClass)

# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=backProplf, verbose = True)


lossMovingAverage =  MovingAverage(10)
norm255 = 1/255
downScaler8c1 = makeDownscaler(d = 8).to(device)
downScaler2c1 = makeDownscaler(d = 2).to(device)

def train(dataloader, epoch):
    print('batchSize:', batchSize)
    maxIters = dataloader.dataset.__len__() / batchSize
    print('TRAINING')
    startTraining = time.time()
    model.train()
    letLastIterEndTime = None

    after_lr = 0
    lossCe = 0
    lossDi = 0
    for i, (x, y, b, c) in enumerate(tqdm(dataloader)):
      # xn = x[0].numpy()
      # yn = y[0].numpy()
      # cv2.imshow('back', xn)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

      # cv2.imshow('back', yn)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      iter = i
      
      # if letLastIterEndTime is not None:
        # print('iterTime =>', i, time.time() - letLastIterEndTime)
      x = x.to(device, dtype=torch.float)
      _, dx, dy = y.shape
      y = y.to(device)
      c = c.to(device)
      b = b.to(device, dtype=torch.float)
      print('b:', b.shape)

      # bd8 = downScaler8c1(c)
      # bd16 = downScaler8c1(c)
      # bd32 = downScaler8c1(c)

      c = c.unsqueeze(1).unsqueeze(1)
      c = c.expand(-1, dx, dy, -1)
      x = torch.abs(x - c)*norm255
      inputTensor = x.permute(0, 3, 1, 2)

      # Compute prediction error
      with torch.autocast(device, enabled=amp):
        predClass, predBBox = model(inputTensor)
        predClass = torch.squeeze(predClass, dim=1)
        predBBox = predBBox.permute(0, 2, 3, 1)
        lossBBoxBatch = lossBbox(pre
                                 
                                 dBBox, c, predClass) 
        lossIter, ce, di = lossClass(predClass, y)
        loss = lossIter / batchSize /accumulate
        lossCe += ce.item() / batchSize /accumulate
        lossDi += di.item() / batchSize /accumulate

      lossAverage = lossMovingAverage.add(loss.item())
      grad_scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

      with torch.no_grad():        
        # maxTarget = torch.max(torch.flatten(y, 1), 1).values
        # print('maxVals:', maxTarget)
        flattentedPred = torch.flatten(pred.detach(), 1 )
        maxVals = torch.max(flattentedPred, 1).values

      if (i%accumulate == 0 and i != 0):
        dataSaver.add({
          'iter': i + maxIters*epoch,
          'di': lossDi,
          'ce': lossCe,
          'lr': after_lr
        })

        lossDi = 0
        lossCe = 0

        before_lr = optimizer.param_groups[0]["lr"]
        grad_scaler.step(optimizer)
        grad_scaler.update()

        optimizer.zero_grad(set_to_none=True)
        after_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        del loss

    torch.save(model, getCheckpointName(modelName, epoch))
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

for epoch in range(epochStart, epochStart + epochs):
    print(f"Epoch {epoch}\n-------------------------------")
    print(f"Epoch {epoch}\n-------------------------------")
    print(f"Epoch {epoch}\n-------------------------------")
    print(f"Epoch {epoch}\n-------------------------------")
    trainDataloader = DataProvider(annotationsDir, batchSize)
    train(trainDataloader, epoch)
    # test(test_dataloader, model, loss_fn)
print("Done!")