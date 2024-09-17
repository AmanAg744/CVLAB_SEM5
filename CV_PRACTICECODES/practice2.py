import cv2 
import numpy as np

img = cv2.imread('book1.jpeg',0)
kernal = np.ones((7,7))/49
blurred_img = cv2.filter2D(img,-1,kernel=kernal)
blurred_img = cv2.GaussianBlur(img,(5,5),1)
mask1 = cv2.subtract(img,blurred_img)
sharpimg = cv2.addWeighted(img,1.4,mask1,-0.4,0)
# blurred_img = cv2.filter2D(sharpimg,-1,kernel=kernal)
mask1 = cv2.subtract(sharpimg,blurred_img)
# mask2 = img - blurred_img

edge = cv2.Canny(sharpimg,150,230)
cv2.imshow('orignal',img)
cv2.imshow('edge',edge)
cv2.imshow('sharpimg',sharpimg)

cv2.waitKey(0)
cv2.destroyAllWindows()