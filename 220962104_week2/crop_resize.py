import cv2 as cv

img = cv.imread('img2.png')
print(img.shape)
cropped_img = img[21:360,131:505]
rsizedimg = cv.resize(img,(360,180),interpolation=cv.INTER_LINEAR)

cv.imshow('original image',img)

cv.imshow('resized image',rsizedimg)
cv.imshow('cropped image',cropped_img)
cv.waitKey(0)
cv.destroyAllWindows()