# Computer_Vision_Toolkit
 ## A Desktop application that provide various computer vision tools like:
## Canny_Edge Detection
 Canny edge detection involves several steps
 - Input image is read and preprocessed using a Gaussian blur to reduce noise, using specified kernel size and sigma value.
 - Then, the gradient magnitude and angle are calculated. Non-maximum suppression is applied to thin the edges, followed by double thresholding to classify potential edge pixels. Weak edges not connected to strong edges are suppressed, and the final edge-detected image is saved. 
 Parameters:
- Gaussian Filter Sigma 
- Gaussian Filter Kernel size
- Upper Threshold
- Lower Threshold

 ![alt text](image.png)

 ## Harris Corner_Detection
 Steps:
1. Read the input image and convert it to grayscale.
2. Apply Gaussian blur to the grayscale image.
3. Calculate the image derivatives 𝐼𝑥Ix and 𝐼𝑦Iy using Sobel 
operators.
4. Compute (𝐼𝑥)^2, (𝐼y)^2, and 𝐼𝑥𝐼𝑦 for each pixel.
5. For each pixel, calculate the sums of squared gradients over the 
window.
6. Construct the matrix M for each pixel.
7. Calculate the corner response function for each pixel.
8. Apply a threshold to the response matrix to identify corners.
9. Mark detected corners on the original image
 Parameters:
- Window size
- Response Threshold
 ![alt text](image-1.png)
 ## Hough Line detection
 Functions:
 - get_edges: Perform edge detection using the Canny edge 
detector.

- Superimpose: draw detected lines on the original image.

- hough_lines:Perform Hough transform to detect lines in the edge detected image.
- detect_lines:Detect lines in an image using the Hough transform and display the result.

 Steps:
1. Read the input image.
2. Get edges using the get_edges function.
3. Detect lines using the hough_lines function.
4. Superimpose detected lines on the original image using the 
superimpose function.
 Parameters:
- threshold (number of lines to detect)
 ![alt text](image-2.png)
 ## Hough ellipse detection
  Steps:
- Convert the input image to grayscale.
- Dynamically adjust the threshold values based on the maximum 
- intensity of the grayscale image.
- Apply the Canny edge detection algorithm using the adjusted 
thresholds.
- Create an empty dictionary named 'accumulator'.
- Print a message indicating that the accumulator has been initialized.
Iterate over non-zero points (y, x) in the edges image using 
np.argwhere() to get the coordinates of these points.
- Nested loops iterate over parameter ranges (a_range, b_range, 
theta_range) to consider various ellipse parameters.
- For each combination of parameters, calculate the center (x_c, y_c) 
of the ellipse using parametric equations.
- Check if the calculated center (x_c, y_c) lies within the boundaries 
of the input image to ensure validity.
- Update the accumulator dictionary with the combination of 
parameters (y_c, x_c, a, b, theta) as the key and increment the 
corresponding count.
- Print a message indicating that the voting process is completed.
- Calculate the threshold value by multiplying half of the 
maximum value in the accumulator dictionary with 0.5.
- Iterate through the items in the accumulator dictionary.
- Filter the items where the value (number of votes) is 
greater than or equal to the threshold.
- Store the keys (representing potential ellipses) of the 
filtered items in the list potential_ellipses.

 Iterate through the list of potential ellipses 
(potential_ellipses).
- For each potential ellipse, scale down the semi-major and 
semi-minor axes using the scale_factor.
- Use the cv2.ellipse function to draw ellipses on the image 
using the scaled parameters.
- Print a message indicating that ellipses have been drawn on the image save_result_image.

 Parameters:
- threshold (number of ellipses to detect)
 ![alt text](image-3.png)
 ![alt text](image-4.png)
 ## Eigen faces dimensionallity reduction
 Steps
 Load, preprocess, and organize images:

Define empty lists images and labels.
Iterate through subject folders in data_path.
Check if the item is a folder.
Iterate through image files in each subject folder.
Read images in grayscale using cv2.imread.
Skip if the image is None.
Optionally resize images.
Flatten and append images to images list.
Append subject folder name to labels list.
Convert lists to numpy arrays faces and labels.
Perform Principal Component Analysis (PCA):

Compute mean face.
Center data by subtracting the mean face.
Apply PCA and fit the model.
Visualize cumulative variance explained by PCA:

Calculate cumulative explained variance.
Plot variance against the number of components.
Reconstruct images with different numbers of principal components:

Initialize figure and subplots.
Display original image.
Reconstruct and plot images for each value of k components.
 ![alt text](image-5.png)
 ## Face Recognition based on eigen faces
 ![alt text](image-6.png)
