import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample2.png')
hsvimg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
b,g,r = cv2.split(img)
histr,bin = np.histogram(r.flatten(),256,(0,255))
def channel_norm_cdf(channel):
    hist,bin = np.histogram(channel.flatten(),256,(0,255))
    cdf = hist.cumsum()
    norm_cdf = cdf *(255/cdf.max())

    return norm_cdf
cdfr = channel_norm_cdf(r)

contstrr = np.interp(r.flatten(),np.arange(256),cdfr).reshape(r.shape)
histr2 = np.histogram(contstrr.flatten(),256,(0,255))
print(contstrr)
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(histr)
plt.subplot(1,3,2)
plt.plot(histr2)


# plt.subplot(1,3,2)
# plt.plot(histg)
# plt.subplot(1,3,3)
# plt.plot(histb)
plt.show()





cv2.imshow('original',img)
# cv2.imshow('hsvimg',hsvimg)

cv2.waitKey(0)
cv2.destroyAllWindows()