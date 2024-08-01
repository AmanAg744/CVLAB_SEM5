import cv2
import cv2 as cv


frame = cv.imread('../week2/image1.jpg')
frame = cv.circle(frame,(506,390),(220),(200,213,48),2)
frame = cv.rotate(frame,cv2.ROTATE_180)
frame = cv2.resize(frame,(1080,720),interpolation=cv2.INTER_LINEAR)
cv.imshow('flower',frame)
# frame2 = cv.imread('image1.jpg',0)
# cv.imwrite('grayscale.jpg',frame2)
# color=frame[20,20]
# for i in range(3):
#     print(int(color[i]))

#v.rectangle(frame,(168,161),(762,618),(48,213,200),2) c
cv.waitKey(0)
cv.destroyAllWindows()