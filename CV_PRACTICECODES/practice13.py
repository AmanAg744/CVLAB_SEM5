import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def hough_transform(image, edge_image, theta_res=1, rho_res=1):
    # Get the dimensions of the image
    rows, cols = edge_image.shape
    
    # Theta values from -90 degrees to +90 degrees
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    
    # Calculate the maximum possible value of rho
    max_rho = int(np.sqrt(rows ** 2 + cols ** 2))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    
    # Create an accumulator array to hold votes for (rho, theta)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # Indices for theta and rho
    rho_idx = lambda r: int(r + max_rho)
    
    # Loop over edge pixels
    for y in range(rows):
        for x in range(cols):
            # If the pixel is an edge
            if edge_image[y, x] > 0:
                # Vote for all theta values
                for theta_idx in range(len(thetas)):
                    theta = thetas[theta_idx]
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    accumulator[rho_idx(rho), theta_idx] += 1

    return accumulator, thetas, rhos

def draw_hough_lines(image, accumulator, thetas, rhos, threshold=100):
    lines_img = np.copy(image)
    
    # Find indices where votes are above the threshold
    rho_idxs, theta_idxs = np.where(accumulator > threshold)
    
    for i in range(len(rho_idxs)):
        rho = rhos[rho_idxs[i]]
        theta = thetas[theta_idxs[i]]
        
        # Convert from polar to Cartesian coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Two endpoints of the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Draw the line
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return lines_img

def plot_hough_space(accumulator, thetas, rhos):
    plt.imshow(accumulator, cmap='hot', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]])
    plt.title('Hough Space')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.colorbar(label='Votes')
    plt.show()

# Load the image and convert it to grayscale
image = cv2.imread('lenna.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Perform Hough Transformation
accumulator, thetas, rhos = hough_transform(image, edges)

# Draw lines based on accumulator votes
lines_img = draw_hough_lines(image, accumulator, thetas, rhos)

# Display the image with detected lines
cv2.imshow('Detected Lines', lines_img)

# Plot the Hough space (accumulator)
plot_hough_space(accumulator, thetas, rhos)

cv2.waitKey(0)
cv2.destroyAllWindows()
