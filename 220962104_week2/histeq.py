import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('sampleimg.jpeg')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
eq_img = cv.equalizeHist(img)
# print(np.max(img))
# print(255/(np.log(np.max(img))))
# c = 255/(np.log(np.max(img)))
# log_transform = c * np.log(1+img)
# log_transform = np.array(log_transform,dtype=np.uint8)

# eq_img = cv.cvtColor(eq_img,cv.COLOR_GRAY2BGR)

# c = 255/(np.log(1+np.max(img)))
# cv.imwrite('equalized_img.jpg',log_transform)
cv.imshow('equalized img function',eq_img)
cv.imshow('original img',img)

cv.waitKey(0)
cv.destroyAllWindows()