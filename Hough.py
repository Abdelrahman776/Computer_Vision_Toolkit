import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
def get_edges(img, min_edge_threshold=100, max_edge_threshold=200):
    # Convert to gray scale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_edge_threshold=0.2*np.max(gray_image)
    max_edge_threshold=0.6*np.max(gray_image)

    # Edge detection on the input image
    edge_image = cv2.Canny(gray_image, min_edge_threshold, max_edge_threshold)
    return edge_image

def hough_lines(src, threshold,resolution=2):
    diagonal = math.ceil(math.sqrt(src.shape[0] * src.shape[0] + src.shape[1] * src.shape[1]))# diagonal length of the image
    
    # Declare the accumulator matrix as zero matrix
    acc = np.zeros((resolution * diagonal, 180), dtype=np.uint8)# resolution*diagonal rows and 180 columns
    # acc matrix is determined by the maximum possible value of the Hough transform. rho = x * cos(theta) + y * sin(theta). The maximum value of rho
    # it occurs when the line passes through the diagonal of the image. By doubling the diagonal length, we ensure that the acc matrix has enough rows to cover all possible lines in the image space.
    lines = []# list to store the detected lines

    # Find the location of edges
    rows, cols = np.nonzero(src)  # returns the indices of the elements that are non-zero.

    cos_theta = np.cos(np.radians(np.arange(-90, 90)))
    sin_theta = np.sin(np.radians(np.arange(-90, 90)))
    for row, col in zip(rows, cols):
        # Calculate the Hough transform for each edge
        rho = np.round(col * cos_theta + row * sin_theta)# rho = x * cos(theta) + y * sin(theta) # rho is a list of rho values as function of theta
        acc[rho.astype(int), np.arange(180)] += 1    # increment the accumulator matrix

    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if acc[i, j] >= threshold:
                lines.append((i, j - 90))# rho and theta in a tuple

    return lines    

def superimpose(img, lines, color):
    src = np.copy(img)
    for i in range(len(lines)):
        rho, theta = lines[i]
        pt1, pt2 = (0, 0), (0, 0) # initialize the points that will be used to draw the line

        cos_theta = math.cos(math.radians(theta))
        sin_theta = math.sin(math.radians(theta))

        x0= rho*cos_theta 
        y0= rho*sin_theta
        
        #pt1 and pt2 are calculated using the polar representation of lines. The formula x = x0 + rho * cos(theta) and y = y0 + rho * sin(theta) gives the endpoint coordinates.
        pt1 = (int(x0 + 1000 * (-sin_theta)), int(y0 + 1000 * (cos_theta)))
        pt2 = (int(x0 - 1000 * (-sin_theta)), int(y0 - 1000 * (cos_theta)))
        cv2.line(src, pt1, pt2, color, 2, cv2.LINE_AA)
    return src

def detect_lines(imgPath, threshold, color=(255, 0, 0)):
    img = cv2.imread(imgPath) 
    edges = get_edges(img)
    lines = hough_lines(edges, threshold,resolution)
    result = superimpose(img, lines, color)
    #  return result
    #  plt.figure("Manually implemented Hough detector")
    #  plt.imshow(result)
    #  plt.xticks([]), plt.yticks([])
    #  plt.show()

detect_lines('sudoku.png',100)
    


