import cv2
import numpy as np

img = cv2.imread('lenna.jpeg')

kernal = np.zeros((5,5))
cent = 5//2
sigma = 1.0
sumval = 0
for i in range(5):
    for j in range(5):
        x = i - cent
        y = j - cent
        kernal[i,j] = (1/(2*np.pi*(sigma**2)) )*np.exp(-((x**2 + y**2)/2*sigma**2))
        sumval += kernal[i,j]

kernal /= sumval

print(kernal)
blurr = cv2.filter2D(img,-1,kernal)

cv2.imshow('img or',img)
cv2.imshow('gaussian ',blurr)
cv2.waitKey(0)
cv2.destroyAllWindows()
