import cv2



def tilePadding(image, sx, sy):
    print(image.shape)
    px = image.shape[0]%sx
    py = image.shape[1]%sy
    print(sx*(image.shape[0]//sx), sy*(image.shape[1]//sy))
    print('px, py', px, py)
    image = cv2.copyMakeBorder(image, 0, px, 0, py, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image

def resizeAndPad(image, symmetric = True, maxSize = None): 
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """

    maxSide = max(*image.shape)
    if maxSize is None:
      maxSize = maxSide

    scale = maxSize/maxSide
       
    # resize image
    x = round(image.shape[0]*scale)
    y = round(image.shape[1]*scale)

    if scale != 1:
      image = cv2.resize(image, [y, x], interpolation = cv2.INTER_AREA)

    # pad image
    xp = (maxSize - x)
    xpl = xp//2
    xpr = xp - xpl

    yp = (maxSize - y)
    ypt = yp//2
    ypb = yp - ypt

    if symmetric:
      image = cv2.copyMakeBorder(image, xpl, xpr, ypt, ypb, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    else:
      image = cv2.copyMakeBorder(image, 0, xp, 0, yp, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image

    print('image:', image.shape)

    cv2.imshow('sample image',image)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() #
    # return image