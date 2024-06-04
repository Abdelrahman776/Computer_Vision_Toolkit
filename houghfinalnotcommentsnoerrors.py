import cv2
import numpy as np
import scipy.ndimage
from scipy.ndimage import convolve
from collections import defaultdict

# def get_edges(img, min_edge_threshold=100, max_edge_threshold=200):
#     # Convert to gray scale
#     min_edge_threshold = 0.2 * np.max(img)
#     max_edge_threshold = 0.6 * np.max(img)
#     # Edge detection on the input image
#     edge_image = cv2.Canny(img, min_edge_threshold, max_edge_threshold)
#     return edge_image

# def initialize_parameters():
#     a_range = (10, 20)
#     b_range = (10, 20)
#     theta_range = np.deg2rad(np.arange(0, 180))
#     print("Parameters set.")
#     return a_range, b_range, theta_range

# def accumulator_filling(edges, a_range, b_range, theta_range, image_shape):
#     accumulator = {}
#     print("Accumulator initialized.")
#     for y, x in np.argwhere(edges):
#         for a in range(*a_range):
#             for b in range(*b_range):
#                 for theta in theta_range:
#                     x_c = int(round(x - a * np.cos(theta)))
#                     y_c = int(round(y + b * np.sin(theta)))
#                     if 0 <= x_c < image_shape[1] and 0 <= y_c < image_shape[0]:
#                         key = (y_c, x_c, a, b, theta)
#                         if key in accumulator:
#                             accumulator[key] += 1
#                         else:
#                             accumulator[key] = 1
#     print("Voting completed.")
#     return accumulator

# def find_draw_ellipses(accumulator, image):
#     threshold = max(accumulator.values()) * 0.5
#     potential_ellipses = [key for key, value in accumulator.items() if value >= threshold]
#     scale_factor = 0.1
#     for y_c, x_c, a, b, theta in potential_ellipses:
#         scaled_a = int(a * scale_factor)
#         scaled_b = int(b * scale_factor)
#         cv2.ellipse(image, (x_c, y_c), (scaled_a, scaled_b), np.degrees(theta), 0, 360, (0, 255, 0), 1)
#     print("Ellipses drawn on image.")
#     return image

# def save_result_image(img, path):
#     cv2.imwrite(path, img)
#     print("Result image saved successfully at:", path)

# def detect_ellipses(imgPath):
#     img = cv2.imread(imgPath)
#     print("Image loaded.")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print("Image converted to grayscale.")
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     print("Gaussian blur applied.")
#     edges = get_edges(blurred)
#     print("Edge detection done.")
#     a_range, b_range, theta_range = initialize_parameters()
#     acc = accumulator_filling(edges, a_range, b_range, theta_range, img.shape)
#     result = find_draw_ellipses(acc, img)
#     save_path = 'D:\oneDrive\Desktop'
#     save_result_image(result, save_path)
#     return save_path
def detect_and_draw_hough_ellipses(image, a_min=30, a_max=100, b_min=30, b_max=100, delta_a=2, delta_b=2, num_thetas=100, bin_threshold=0.4, min_edge_threshold=100, max_edge_threshold=150):
        """
        Detect ellipses using Hough Transform.
        Args:
            image_path (str): Path to the input image file.
            a_min (int): Minimum semi-major axis length of ellipses to detect.
            a_max (int): Maximum semi-major axis length of ellipses to detect.
            b_min (int): Minimum semi-minor axis length of ellipses to detect.
            b_max (int): Maximum semi-minor axis length of ellipses to detect.
            delta_a (int): Step size for semi-major axis length.
            delta_b (int): Step size for semi-minor axis length.
            num_thetas (int): Number of steps for theta from 0 to 2PI.
            bin_threshold (float): Thresholding value in percentage to shortlist candidate ellipses.
            min_edge_threshold (int): Minimum threshold value for edge detection.
            max_edge_threshold (int): Maximum threshold value for edge detection.
        Returns:
            tuple: A tuple containing the output image with detected ellipses drawn and a list of detected ellipses.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edge_image = cv2.Canny(
            gray_image, min_edge_threshold, max_edge_threshold)

        # Get image dimensions
        img_height, img_width = edge_image.shape[:2]

        # Initialize parameters for Hough ellipse detection
        dtheta = int(360 / num_thetas)
        thetas = np.arange(0, 360, step=dtheta)
        as_ = np.arange(a_min, a_max, step=delta_a)
        bs = np.arange(b_min, b_max, step=delta_b)
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))
        ellipse_candidates = [(a, b, int(a * cos_thetas[t]), int(b * sin_thetas[t]))
                              for a in as_ for b in bs for t in range(num_thetas)]

        # Initialize accumulator
        accumulator = defaultdict(int)

        # Iterate over each pixel and vote for potential ellipse centers
        for y in range(img_height):
            for x in range(img_width):
                if edge_image[y][x] != 0:
                    for a, b, acos_t, bsin_t in ellipse_candidates:
                        x_center = x - acos_t
                        y_center = y - bsin_t
                        accumulator[(x_center, y_center, a, b)] += 1

        # Initialize output image
        output_img = image.copy()

        # Store detected ellipses
        out_ellipses = []

        # Loop through the accumulator to find ellipses based on the threshold
        for candidate_ellipse, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            x, y, a, b = candidate_ellipse
            current_vote_percentage = votes / num_thetas
            if current_vote_percentage >= bin_threshold:
                out_ellipses.append((x, y, a, b, current_vote_percentage))

        # Perform post-processing to remove duplicate ellipses
        pixel_threshold = 5
        postprocess_ellipses = []
        for x, y, a, b, v in out_ellipses:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(a - ac) > pixel_threshold or abs(b - bc) > pixel_threshold for xc, yc, ac, bc, v in postprocess_ellipses):
                postprocess_ellipses.append((x, y, a, b, v))
        out_ellipses = postprocess_ellipses

        # Draw detected ellipses on the output image
        for x, y, a, b, v in out_ellipses:
            output_img = cv2.ellipse(
                output_img, (x, y), (a, b), 0, 0, 360, (0, 255, 0), 2)

        return output_img, out_ellipses
if __name__ == "__main__":
    # save_path = detect_ellipses()
    # result_image = cv2.imread(save_path)
    image=cv2.imread('D:\oneDrive\Desktop\IMAGE\qq.jpg')
    result_image,r=detect_and_draw_hough_ellipses(image)
    cv2.imshow('Detected Ellipses', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Finished displaying results.")