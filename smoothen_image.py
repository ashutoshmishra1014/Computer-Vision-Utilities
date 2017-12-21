import cv2
import numpy as np

# input file name of the file to be smoothed
image = cv2.imread('difficult.jpg')
blur = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
#use fastNlMeansDenoising for gray scale images, several other alternatives available
blur = cv2.GaussianBlur(blur,(15,15),0)
# output file name of the smoothed file
cv2.imwrite('smooth.png', blur)
# a colored smoothed image will be created