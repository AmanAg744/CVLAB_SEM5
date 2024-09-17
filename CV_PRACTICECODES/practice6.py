import cv2
import numpy as np
img = cv2.imread('beans.jpg', 0)

# Find the min and max pixel values
min_val = np.min(img[np.nonzero(img)])
max_val = np.max(img)

# Stretch the pixel values to the full range [0, 255]
stretched_img = ((img - min_val) / (max_val - min_val)) * 255
stretched_img = np.clip(stretched_img, 0, 255).astype(np.uint8)

# Show the result
cv2.imshow('Original Image', img)
cv2.imshow('Contrast Stretched Image', stretched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
