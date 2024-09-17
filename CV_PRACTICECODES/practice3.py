import cv2 
import numpy as np

img = cv2.imread('book1.jpeg',0)
# blurred_img = cv2.GaussianBlur(img,(5,5),1)
kernalx = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]],dtype=np.int8)

sobelx = cv2.filter2D(img,-1,kernalx)
# kernaly = np.array([[-1,-2,-1],
#                     [0,0,0],
#                     [1,2,1]],dtype=np.int8)

# sobel_y = cv2.filter2D(img,-1,kernaly)

# mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
# mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
# mag = mag.astype(np.int8)
sobel_x = cv2.Sobel(img,-1,1,0,ksize=3)
cv2.imshow('xgradient',sobel_x)
cv2.imshow('manual',sobelx)

cv2.waitKey(0)
cv2.destroyAllWindows()