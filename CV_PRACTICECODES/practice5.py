import cv2
import numpy as np
import matplotlib.pyplot as plt



img = cv2.imread('beans.jpg')
# hist = cv2.calcHist(img,[0],None,[256],[0,256]).flatten()
hist1,bin = np.histogram(img.flatten(),256,[0,256])
cdf1 = hist1.cumsum()
# cdf_min = cdf.min()

# cdf_normalized = (cdf - cdf_min) * 255  / (cdf.max() - cdf.min())
# # eqimg = np.interp(img,np.arange(256),cdf_normalized).reshape(img.shape)
# cdf_normalized = cdf_normalized.astype(np.uint8)
# print(cdf_normalized)
# eqimg = cdf_normalized[img]
r1 = 127
r2 = 133
eqimg = np.copy(img)
eqimg = eqimg.astype(np.uint8)
mask = (img >= r1) & (img<=r2)
hist,bins = np.histogram(img[mask].flatten(),r2-r1 + 1,[r1,r2+1])
cdf = hist.cumsum()
cdf_min = cdf.min()

cdf_normalized = (cdf - cdf_min) * 255  / (cdf.max() - cdf.min())
cdf_normalized = cdf_normalized.astype(np.uint8)
print(cdf_normalized)
eqimg[mask] = cdf_normalized[img[mask]-r1]



# gamma = [0.1,0.8,1,1.2,3]
# for g in gamma:
#     gammaimg = (((img/255) ** g) * 255)
#     gammaimg = gammaimg.astype(np.uint8)
#     cv2.imshow('EqualizedImg',gammaimg)
#     cv2.waitKey(0)



plt.figure(figsize=(12,8))
plt.plot(cdf1)
plt.show()
cv2.imshow('original',img)
cv2.imshow('EqualizedImg',eqimg)
cv2.waitKey(0)
cv2.destroyAllWindows()