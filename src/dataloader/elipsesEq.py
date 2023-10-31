import cv2
import math
import random
import numpy as np
from datetime import datetime
from numba import njit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# random.seed(datetime.now().timestamp())

sizeNoiseDelta = 0.2
@njit
def genElipsesParams(dxCenter, dyCenter, dlMax, dlMin, dxyNoise = 1, arSize = 0.5):
  theta = 2*math.pi*random.random()
  x0 = dxCenter + random.random()*dxyNoise
  y0 = dyCenter + random.random()*dxyNoise
  cosTheta = math.cos(theta)
  sinTheta = math.sin(theta)
  
  # we only need to scale 1 axis, the rotation will do the flipping
  # we keep 1 lengths constant to try to control the minimum and maximum sizes
  aspectRatio = random.random()*(1-arSize) + arSize

  # we add a little bit of size noise

  sizeNoise = random.random()*sizeNoiseDelta + (1 - sizeNoiseDelta)

  a = (aspectRatio*(sizeNoise*(dlMax - dlMin)/2 + dlMin/2))**2
  b = ((sizeNoise*(dlMax - dlMin)/2 + dlMin/2))**2
  return x0, y0, cosTheta, sinTheta, a, b

@njit
def inElipses(dx, dy, _, __, cosTheta, sinTheta, ar, br):
  xr = (dx*cosTheta - dy*sinTheta)**2
  yr = (dx*sinTheta + dy*cosTheta)**2
  return ((xr/ar) + (yr/br))

@njit
def noisyColor(c, s):
  color = random.gauss(c, s)
  color = max(color, 0)
  color = min(color, 255)
  return color

@njit
def gaussian2D(x, y, x0, y0, sx, sy, a):
  dx2=(x-x0)**2
  dy2=(y-y0)**2
  exp=dx2/sx + dy2/sy
  return a*np.exp(-exp)

@njit
def bightnessColorFactor():
  r = random.random()
  g = random.random()
  b = random.random()
  norm = math.sqrt(r**2 + g**2 + b**2)
  r = (r/norm)*0.2 + 0.9
  g = (g/norm)*0.2 + 0.9
  b = (b/norm)*0.2 + 0.9
  return [r, g, b]

@njit
def sigmoid(x, sig = 1):
  if x > 10:
    return 0
  
  denominator = 1 + np.exp((x - 1)*sig)
  v = 1 / denominator
  return v

@njit
def bbox(x0, y0, cosTheta, sinTheta, a, b):
  c1 = math.sqrt(a*(cosTheta**2) + b*(sinTheta**2))
  c2 = math.sqrt(a*(sinTheta**2) + b*(cosTheta**2))
  px1 = x0 + c1
  px2 = x0 - c1
  py1 = y0 + c2
  py2 = y0 - c2

  return min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)

@njit
def bboxWH(_, __, cosTheta, sinTheta, a, b):
  px1 = math.sqrt(a*(cosTheta**2) + b*(sinTheta**2))
  px2 = math.sqrt(a*(cosTheta**2) + b*(sinTheta**2))
  py1 = math.sqrt(a*(sinTheta**2) + b*(cosTheta**2))
  py2 = math.sqrt(a*(sinTheta**2) + b*(cosTheta**2))

  return abs(px1 + px2), abs(py1 + py2)



@njit
def propDensity(dxe, dye, paramsBrighness, sig):
  val = inElipses(dxe, dye, *paramsBrighness)
  prop = sigmoid(val, sig)
  return prop, val

minBrightness = 0.5
maxBrightness = 1.5

@njit
def printElipses(imagePatch, mask, bbox, color, l, paramsElipse, saveMask, vx, vy):
  r, g, b = color
  [dx, dy, _] = imagePatch.shape
  image = imagePatch

  # # Random pixel noise
  rgbS = math.sqrt(random.random()*15) 
  
  # reflection noise
  brigtnessBlur = math.sqrt(l*(0.0001 + random.random()*0.005))
  brightLocX = dx*random.random() - dx/2
  brightLocY = dy*random.random() - dy/2
  paramsBrighness = genElipsesParams(brightLocX, brightLocY, l*1, l*0.1, 0.1)

  #gaussian for impainting amplitude
  a = random.random()*0.5 + 1
  # sigmaBrighness = 2*random.random()*imageBlur

  amplitudeBrighness = random.random()*(maxBrightness - minBrightness)
  # birghtnessSign = round(random.random())*2 - 1

  bcfr, bcfg, bcfb = bightnessColorFactor()
  xmin = dx
  ymin = dy
  xmax = 0
  ymax = 0
  
  sig = random.random()*7 + 3
  xc = paramsElipse[0]
  yc = paramsElipse[1]
  bW, bH = bboxWH(*paramsElipse)
  
  v = (vx**2 + vy**2)**0.5
  steps = math.ceil(v/l/0.025)
  stepsInt = steps+1

  for i in range(dx):
    for j in range(dy):
      x = i+0.5
      y = j+0.5
      dxe = (x - xc)
      dye = (y - yc)

      # prop, val = propDensity(dxe, dye, paramsElipse, sig)

      # print(val, prop)
      # print('vx =>', vx, dxe, dye)
      prop = []
      val = []
      for l in range(steps):
        dxv = vx * l/steps
        dyv = vy * l/steps
        dxt = dxe - dxv
        dyt = dye - dyv

        if (i - dxv) < 0 or (j - dyv) < 0:
          continue
        # print('====', dyt, dxt, dxt == dxe and dyt == dye)
        # if dxt < 0 or dyt < 0 :
        propT, valT = propDensity(dxt, dyt, paramsElipse, sig)

        prop.append(propT)
        val.append(valT)
      

      valCenter = inElipses(dxe - vx/2, dye - vy/2, *paramsElipse)

      val = areaIntegralFromArray(val, 1/stepsInt)
      prop = areaIntegralFromArray(prop, 1/stepsInt)
      

      # val = val*(1/steps)
      # prop = prop*(1/steps)
      
      # print('prop:', prop, val)  
      
      # print(val==valE and prop == propE)

      valBright = inElipses(dxe, dye, *paramsBrighness)
      propBright = amplitudeBrighness*np.exp(-(valBright**2)*brigtnessBlur) + minBrightness

      # R = noisyColor(r*propBright*bcfr, rgbS)*prop + image[i, j, 0]*(1 - prop)
      # G = noisyColor(g*propBright*bcfg, rgbS)*prop + image[i, j, 1]*(1 - prop)
      # B = noisyColor(b*propBright*bcfb, rgbS)*prop + image[i, j, 2]*(1 - prop)

      R = noisyColor(r*propBright*bcfr, rgbS)*prop + image[i, j, 0]*(1 - prop)
      G = noisyColor(g*propBright*bcfg, rgbS)*prop + image[i, j, 1]*(1 - prop)
      B = noisyColor(b*propBright*bcfb, rgbS)*prop + image[i, j, 2]*(1 - prop)
      image[i, j] = [round(R), round(G), round(B)]
      
      if valCenter < 1 and saveMask:
        bbox[i, j] = [
          dxe - vx/2,
          dye - vy/2,
          bW,
          bH 
        ]
        mask[i, j] = 1
        


  return [xmin, ymin, xmax, ymax]

dt = 1
maxDotSizeToPatchProportion = 0.6
minDotSizeToPatchProportion = 0.4

@njit
def genPatchFeatures(dx, dy):
  dl = random.randint(math.floor(dx/10), math.ceil(dx/5))
  v0 = dl*random.random()*0.5

  vxyr = random.random()*math.pi/2
  
  vx = v0*math.sin(vxyr)
  vy = v0*math.cos(vxyr)

  xLimit = round(dx - dl - vx*dt)
  yLimit = round(dy - dl - vy*dt)

  x0 = random.randint(0, xLimit)
  y0 = random.randint(0, yLimit)

  xf = x0 + vx*dt + dl
  yf = y0 + vy*dt + dl

  return round(x0), round(y0), round(xf), round(yf), dl, vx, vy

@njit
def elipsesIntersect(e1, e2):
  dx = (e1[0]-e2[0])**2
  dy = (e1[1]-e2[1])**2
  d = math.sqrt(dx + dy)
  
  e1r = e1[2]
  e2r = e2[2]

  return d < e1r + e2r

def intersectsPreviousElipses(prevElipses, newElipses):
  intersects = False
  i = 0
  while not intersects and len(prevElipses) > i:
    intersects = elipsesIntersect(prevElipses[i], newElipses)
    i += 1

  return intersects
@njit
def intersectionBboxes(boxA, boxB):
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1]);
  xB = min(boxA[2], boxB[2]);
  yB = min(boxA[3], boxB[3]);

  return max(0, xB - xA) * max(0, yB - yA);


def intersectionPreviousBboxes(previousBboxes, newBboxes):
  return

@njit
def getRandomColor():
  r = random.randint(0, 255)
  g = random.randint(0, 255)
  b = random.randint(0, 255)
  return r, g , b

@njit
def furthersColor(r, g, b):
  return max(r, 255 - r), max(g, 255 - g), max(b, 255 - b)
  
@njit
def colorDistance(c1, c2):
  r1, g1, b1 = c1
  rm, gm, bm = furthersColor(*c1)
  r2, g2, b2 = c2
  
  n = math.sqrt((rm)**2 + (gm)**2 + (bm)**2)
  d = math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
  return d/n

@njit
def areaIntegralTrapezoid(h1, h2, d):
  hMin = min(h1, h2)
  hMax = max(h1, h2)
  return d*hMin + d*(hMax - hMin)/2

@njit
def areaIntegralFromArray(arr, d):
  totArea = 0
  if len(arr) == 1:
    totArea = arr[0]

  for i in range(len(arr)-1):
    totArea += areaIntegralTrapezoid(arr[i], arr[i+1], d)
  return totArea

def drawElipses(imageData):
  image = np.copy(imageData)
  mask = np.zeros((image.shape[0], image.shape[1]))
  bArr = np.zeros((image.shape[0], image.shape[1], 4))
  [dx, dy, _] = image.shape

  n = random.randint(5, 30)
  n = 30
  mainColor = getRandomColor()
  elipsesData = []
  bboxes = []
  newBoxes = []

  for i in range(n):
    color = mainColor
    xmin, ymin, xmax, ymax, dl, vx, vy = genPatchFeatures(dx, dy)
    newBoxes.append([xmin, ymin, xmax, ymax])

    # ymax = ymin + dl
    # xmax = xmin + dl

    patch = image[xmin:xmax, ymin:ymax]
    # [dpx, dpy, _] = patch.shape

    # xdStart = dl/2
    # ydStart = dl/2
    # xdEnd = (xmax-dl)/2
    # ydEnd = (ymax-dl)/2


    elipsesParams = genElipsesParams(
      dl/2,
      dl/2,
      dl*maxDotSizeToPatchProportion,
      dl*minDotSizeToPatchProportion
    )
    x0, y0, cosTheta, sinTheta, a, b = elipsesParams

    # newElipse = [xmin + x0, ymin + y0, math.sqrt(max(a, b))]
    newElipse = [xmin + x0 + vx/2, ymin + y0 + vy/2, math.sqrt(max(a, b))]
    intersects = False
    if (len(elipsesData) > 0):
      intersects = intersectsPreviousElipses(elipsesData, newElipse)

    if not intersects:
      elipsesData.append(newElipse)

      maskPatch = mask[xmin:xmax, ymin:ymax]
      bboxPatch = bArr[xmin:xmax, ymin:ymax]

      dummy = random.random()
      saveMask = True
      if dummy  < 0.25 and i > 0:
        newColor = getRandomColor()
        distance = colorDistance(newColor, color)
        if distance > 0.1:
          saveMask = False
          color = newColor

      printElipses(patch, maskPatch, bboxPatch, color, dl, elipsesParams, saveMask, vx, vy)

      if saveMask:
        bboxes.append(bbox(x0 + xmin, y0 + ymin, cosTheta, sinTheta, a, b))

  return image, mask, bArr, mainColor, bboxes, newBoxes


