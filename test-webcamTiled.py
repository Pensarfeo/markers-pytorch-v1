
# import the opencv library
import cv2
import time
import addSrcToPath
import torch

# import numpy as np
# import os
# import random
from model.yolov8.semantic_detection_d0 import Model as YoloV8
# from dataloader import DataProvider
# import cv2
from utils.math import MovingAverage
from utils.tensor import tile, untile
from utils.image import tilePadding
# random.seed(time.time())

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
print('purpleTensor:', purpleTensor.shape)

# model = torch.load('./outputs/Yolov8-cocoTest-noise-brighness-colortwicking-counterExamples-d0-test/epochs-5.pt')
model = torch.load('./outputs/Yolov8-cocoTest-noise-brighness-colortwicking-counterExamples-d0-test-test-crl-asym/epochs-4.pt')
model.eval()
print(next(model.parameters()).is_cuda, next(model.parameters()).is_cuda)

purple = [98, 68, 158]
purple = [122, 92, 182]
# purple = [107, 90, 203]
tileDx = 256
tileDy = 256
  
# define a video capture object

fps = MovingAverage(20)

vid = cv2.VideoCapture(0)
startTime = time.time()



with torch.no_grad():
  while(True):
    dtA = fps.add(1/(time.time() - startTime + 0.00000000001))
    print('fps =>', dtA)
    startTime = time.time()
        
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = tilePadding(frame - purple, tileDx, tileDy)
    image = torch.from_numpy(image)
    image = image.to(device, dtype=torch.float)
    image = torch.abs(image)/255
    
    inputStacked, tilingInfo = tile(image)
    imageUnstacked = untile(inputStacked, tilingInfo)
    # Display the resulting frame
    
    input = inputStacked.permute(0, 3, 1, 2)
    pred, dxdy, wh, __ = model(input)
    pred = pred.squeeze()
    pred = torch.sigmoid(pred)

    output = untile(pred, tilingInfo)
    output = output.cpu().numpy()
    
    # cv2.imshow('frame', imageUnstacked.cpu().numpy())
    cv2.imshow('frame', output)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
  # After the loop release the cap object
  vid.release()
  # Destroy all the windows
  cv2.destroyAllWindows()