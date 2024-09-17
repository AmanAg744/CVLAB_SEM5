import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread('book1.jpeg', 0)  # Change path to your image

# Step 1: Apply Gaussian Blur to reduce noise
# Define a Gaussian kernel (5x5 kernel with sigma=1)
gaussian_kernel = cv2.getGaussianKernel(5, 1)
gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # Convert 1D kernel to 2D
blurred_img = cv2.filter2D(img, -1, gaussian_kernel)

# Step 2: Sobel Operator to detect edges in x and y directions
# Define Sobel kernels
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])  # Horizontal edge detection

sobel_y_kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])  # Vertical edge detection

# Apply Sobel filtering
sobel_x = cv2.filter2D(blurred_img, -1, sobel_x_kernel).astype(np.float32)  # Gradient in X direction
sobel_y = cv2.filter2D(blurred_img, -1, sobel_y_kernel).astype(np.float32)  # Gradient in Y direction

# Step 3: Compute gradient magnitude
magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.float32)

# Step 4: Normalize to 0-255 for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convert magnitude to uint8
magnitude = magnitude.astype(np.uint8)

# Step 5: Apply a threshold to detect strong edges
# (This part mimics Canny's non-maximum suppression to some extent)
_, thresholded_edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

# Display the resulting edge detection
cv2.imshow('Detected Text Edges', thresholded_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
