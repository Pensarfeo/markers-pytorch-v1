import torch
import glob
import os

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
      # m.weight.data.normal_(0, 0.01)
      # torch.torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
      torch.torch.nn.init.xavier_normal_(m.weight.data)
      if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

    elif isinstance(m, torch.nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()

def getCheckpoints(dirPath):
    return 

def getCheckpoint(dirPath, n = -1):
    checkpointPaths = glob.glob(os.path.join(dirPath, '*.pt'))   
    epochs = []

    for f in checkpointPaths:
        number = int(os.path.basename(f)[7:9].replace('.', ''))
        epochs.append(number)

    match = n
    if n == -1 and len(epochs) > 0:
        match = max(epochs)

    if len(epochs) == 0 or not match in epochs :
      return -1, ''

    maxEpochIndx = epochs.index(match)

    return match, checkpointPaths[maxEpochIndx]
    
def populateModel(model, checkpoint):
  for keyPath in checkpoint['model_state_dict'].keys():
    # print('---', keyPath, '---')
    savedTensor = checkpoint['model_state_dict'][keyPath]
    ele2 = model

    pathKeys = keyPath.split('.') 

    for kp in pathKeys:
      ele2 = getattr(ele2, kp)

    targetTensorShape = ele2.shape
    initiaShape = savedTensor.shape
    expandedWeights = savedTensor
    for i, d in enumerate(targetTensorShape):
      ratio = d // initiaShape[i]
      if ( ratio > 1):
        toConcat = [expandedWeights for i in range(ratio)]
        expandedWeights = torch.concat(toConcat, dim=i)
    ele2.data = expandedWeights

def getModelSize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('=> model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

def initialize(model, path, cache = True, n = -1):
  epoch = -1

  if cache:
    epoch, checkpointPath = getCheckpoint(path, n)
     
  if (epoch < 0) and cache:
     print('@@@@@', f'No checkpoint found for epoch {n}')
     cache = False
     

  if cache:
      print('@@@@@', f'initializing model from checkpoint {epoch}')
      checkpoint = torch.load(checkpointPath)
      populateModel(model, checkpoint)
  else:
      print('@@@@@', f'Initializing model from scratch')
      checkpoint = None
      initialize_weights(model)

  getModelSize(model)
  
  return epoch, checkpoint
