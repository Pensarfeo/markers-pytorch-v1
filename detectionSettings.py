import math
minRealHeight = 1.4
minPixelHeight = 1080
maxRealHeight = 2
maxPixelHeight = 1920

crankLength = 0.175
maxRPS = 100/60
dotSize = 0.01
exposureTime = 1/(120)

totRadius = 2*math.pi*crankLength
lenthTraveledPS = totRadius * maxRPS
lengthTraveledInFrame = lenthTraveledPS*exposureTime
print('max distance (m) traveled =>', lengthTraveledInFrame)
fuzydotSize = dotSize + lengthTraveledInFrame

traveledArcDefs = 360 * exposureTime
print('max arc traveled =>', traveledArcDefs)
traveledArc = 2*math.pi * exposureTime

maxSpeed =  lengthTraveledInFrame/exposureTime
print('max speed verctor length =>', maxSpeed)

# lengthTraveledInFrame
# There is a neglegible difference between the arc lenth and the straight lengt of about 0.03%
# this is the equation => math.sin(traveledArc/2)/(math.pi*exposureTime)
arcTravelledLinePathRatio = math.sin(traveledArc/2)/(math.pi*exposureTime)
print('straght path / arc path => ', arcTravelledLinePathRatio)

minMeterToPixel = minPixelHeight/minRealHeight
maxMeterToPixel = maxPixelHeight/minRealHeight

expectedDodPixelSizes = []
expectedWindowPixelSizes = []

for rh in [minRealHeight, maxRealHeight]:
    for ph in [minPixelHeight, maxPixelHeight]:
        cmToPixel = (ph/rh)*0.01
        expectedDodPixelSizes.append(cmToPixel*1)
        expectedDodPixelSizes.append(cmToPixel*2)
        expectedDodPixelSizes.append(cmToPixel*10)

        print('Real Height', rh*1.0, 'pixel heith', ph, 'Pixel Diameter', cmToPixel*1, cmToPixel*2, 'windowSize',  cmToPixel*10, cmToPixel*5)


# the tolerated marker pixel size go from 5px 25px
# the window size will always be 10x to 5x the marker size
# with a looking window size of 128x128, this is equal to a patch side of
[128/10, 128/5] 