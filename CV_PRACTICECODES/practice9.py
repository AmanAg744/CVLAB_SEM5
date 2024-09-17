import cv2 as cv
import numpy as np

def canny_edge(img, low_th=50, high_th=150):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv.GaussianBlur(gray, (3, 3), 1.4)
    
    # Compute Sobel gradients in x and y directions
    sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and direction of gradients
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Use arctan2 for correct angle
    
    # Normalize directions to [0, 180)
    direction[direction < 0] += 180
    
    # Initialize suppression result
    supp = np.zeros_like(magnitude, dtype=np.float32)
    
    # Non-maximum suppression
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            angle = direction[i, j]
            
            q = 255
            r = 255
            
            # Check the direction and apply non-maximum suppression
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                supp[i, j] = magnitude[i, j]
    
    # Thresholding (Double threshold)
    strong_edges = (supp > high_th)
    weak_edges = (supp >= low_th) & ~strong_edges
    
    # Output edge map
    edges = np.zeros_like(supp, dtype=np.uint8)
    edges[strong_edges] = 255
    
    # Edge tracking by hysteresis: connecting weak edges to strong edges
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = 255
    
    return edges

# Load an image
img = cv.imread('book1.jpeg')

# Run the Canny edge detector
edges = canny_edge(img)

# Display the result
cv.imshow('Edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()