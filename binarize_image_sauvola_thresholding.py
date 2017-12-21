import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


matplotlib.rcParams['font.size'] = 9
#input image file name
image = cv2.imread('smooth.png',0)
kernel = np.ones((5,5),np.uint8)
#dilate will make the letters finer, may be not so pleasing looking, choose to apply it depending on the results you get
image = cv2.dilate(image,kernel,iterations = 1)
binary_global = image > threshold_otsu(image)
window_size = 25
thresh_sauvola = threshold_sauvola(image, window_size=window_size)
binary_sauvola = image > thresh_sauvola
plt.figure()
img = binary_sauvola
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.title('Sauvola Threshold')
plt.axis('off')
#plt.show()
#output filename here, the binarized image
plt.savefig('binary_sauvola.png', dpi=1500)