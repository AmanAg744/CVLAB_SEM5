import cv2
import cv2 as cv
import numpy as np

img = cv2.imread('fruits.jpg')

kernal2 = np.ones((5,5),dtype=np.float32) / 25

filtimg = cv.filter2D(src = img,ddepth=-1,kernel=kernal2 )
mask = img - filtimg
sharpedimg = cv.addWeighted(img,1,mask,-0.035,0)
cv.imshow('OI',img)
# cv.imshow('FI',filtimg)
cv.imshow('SHARP IMAGE ',sharpedimg)
cv.waitKey(0)
cv.destroyAllWindows()