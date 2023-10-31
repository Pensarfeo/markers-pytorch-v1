
# import the opencv library
import cv2
import time

# import numpy as np
# import os
# import random
import torch
from model.yolov8.semantic import Model
# from dataloader import DataProvider
# import cv2
from utils.image import pad as padImage
from utils.math import MovingAverage
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

# annotationsDir = os.path.join('.', 'data', 'coco', 'val2017')
# trainDataloader = DataProvider(annotationsDir, 1)

model = torch.load('../Yolov8_bz4_Epochs1_randxy_n-5_dl-50_center_color_epochs-9.pt')
model.eval()
  
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

    image = torch.from_numpy(padImage(frame))
    image = image.to(device, dtype=torch.float)
    image = torch.abs(image - purpleTensor)/255
    inputStacked, ntx, nty = tile(image)
    print('image.shape:', image.shape)

    # Display the resulting frame
    
    input = inputStacked.permute(0, 3, 1, 2)
    print('inputStacked.shape:', input.shape)
    pred = model(input)
    pred = pred.squeeze()
    pred = torch.sigmoid(pred)
    
    output = untile(inputStacked, ntx, nty)
    print('output:=>', output.shape)
    output = output.cpu().numpy()
      
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