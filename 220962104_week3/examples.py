import cv2
import cv2 as cv
import numpy as np

img = cv.imread('lena.jpeg')
cv2.imshow('Original Image',img)

GB = cv.GaussianBlur(img,(7,7),0)
cv.imshow('GB',GB)

MB = cv.medianBlur(img,5)
cv.imshow('MB',MB)

bilate = cv.bilateralFilter(img,9,75,75)
cv.imshow('BF',bilate)


cv.imshow('Unsharp M',img-MB)
cv2.waitKey(0)
cv.destroyAllWindows()