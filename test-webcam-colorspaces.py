
# import the opencv library
import cv2
import time
import addSrcToPath
import numpy as np
# import os
# import random
from utils.math import MovingAverage
# random.seed(time.time())

purple = [158, 68, 98]

# define a video capture object

fps = MovingAverage(20)

vid = cv2.VideoCapture(0)
startTime = time.time()
while(True):
  dtA = fps.add(1/(time.time() - startTime + 0.00000000001))
  print('fps =>', dtA)
  startTime = time.time()
      
  # Capture the video frame
  # by frame
  ret, frame = vid.read()
  imgHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 0]
  cv2.imshow('frame', imgHLS)

  # colorCentered = np.abs(frame - purple)/255
  # cv2.imshow('frame', colorCentered)
  # Display the resulting frame
    
  # the 'q' button is set as the
  # quitting button you may use any
  # desired button of your choice
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()