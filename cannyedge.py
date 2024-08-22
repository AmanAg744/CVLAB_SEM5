import cv2
import cv2 as cv
import numpy as np
def canny_edge(img,low_th = 30,high_th = 150):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray,(5,5),1.4)
    sobel_x = cv.Sobel(blurred,cv.CV_64F,1,0,ksize=3)
    sobel_y = cv.Sobel(blurred,cv.CV_64F,0,1,ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y**2)
    direction  = np.arctan(sobel_y,sobel_x)*(180/np.pi)
    direction[direction < 0] += 180
    direction = np.round(direction/45) * 45
    supp = np.zeros_like(gray)
    for i in range(1,gray.shape[0] - 1):
        for j in range(1,gray.shape[1] - 1):
            angle = direction[i,j]
            q = 255
            r = 255

            if(0<=angle<45) or (angle>=135):
                q = magnitude[i,j + 1]
                r = magnitude[i,j - 1]
            elif(45 <= angle <135):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (135 <= angle < 180) or (angle < 0):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                supp[i, j] = magnitude[i, j]
    strong_edges = (supp>high_th)
    weak_edges = (supp>=low_th) & ~strong_edges

    edges  =np.zeros_like(supp,dtype=np.uint8)
    edges[strong_edges] = 255

    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = 255
    return edges

img = cv.imread('book1.jpeg')
# low_thresholds = [30, 50, 70]
# high_thresholds = [100, 150, 200]
#
# for low in low_thresholds:
#     for high in high_thresholds:
#         edges = canny_edge(img, low_th=low, high_th=high)
#         cv.imshow('Edges',edges)
#         cv2.waitKey(0)
edges = canny_edge(img)
cv.imshow('Edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()



