import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the data
data_path = 'preprocessing'

# Initialize lists to hold images and labels
images = []
labels = []

# Loop over each subject folder
for subject_folder in sorted(os.listdir(data_path)):
    subject_path = os.path.join(data_path, subject_folder)
    if os.path.isdir(subject_path):
        for image_name in sorted(os.listdir(subject_path)):
            image_path = os.path.join(subject_path, image_name)
            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            # Resize the image to a smaller size (e.g., 50x50)
            image = cv2.resize(image, (50, 50))
            # Flatten the image
            images.append(image.flatten())
            labels.append(subject_folder)

# Convert list to numpy array
faces = np.array(images)
labels = np.array(labels)

# Step 1: Mean face
mean_face = np.mean(faces, axis=0)

# Step 2: Center the data
centered_faces = faces - mean_face

# Step 3: Compute the covariance matrix using matrix multiplication for efficiency
cov_matrix = np.dot(centered_faces.T, centered_faces) / centered_faces.shape[0]

# Ensure covariance matrix is symmetric
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# Step 4: Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ensure all eigenvalues are non-negative (due to numerical issues, very small negative values may appear)
eigenvalues = np.maximum(eigenvalues, 0)

# Step 5: Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Take the real part of the eigenvectors (to handle complex numbers)
eigenvectors = np.real(eigenvectors)

# Explained variance
explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# Reconstruct images using different k values
k_values = [5, 10, 30, 60]
fig, axs = plt.subplots(len(k_values), 2, figsize=(10, 10))

# Original image (first image from the dataset)
original_image = faces[0].reshape((50, 50))
# Plot original image in the first row
plt.subplot(len(k_values) + 1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot explained variance for the original image (all components)
plt.subplot(len(k_values) + 1, 2, 2)
plt.plot(explained_variance)
# plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.title('Variance (Original)')

# Reconstructions
for i, k in enumerate(k_values):
    top_k_eigenvectors = eigenvectors[:, :k]
    transformed = np.dot(centered_faces, top_k_eigenvectors)
    reconstructed = np.dot(transformed, top_k_eigenvectors.T) + mean_face
    reconstructed_image = np.real(reconstructed[0].reshape((50, 50)))

    # Plot reconstructed image
    plt.subplot(len(k_values) + 1, 2, 2 * (i + 1) + 1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'{k} Components')
    plt.axis('off')

    # Plot variance explained up to k components
    plt.subplot(len(k_values) + 1, 2, 2 * (i + 1) + 2)
    plt.bar(range(1, k + 1), explained_variance[:k])
    plt.ylim(0, 1)
    # plt.xlabel('Number of Components')
    plt.ylabel('Variance')
    plt.title(f'Variance: {explained_variance[k - 1]:.2f}')

plt.tight_layout()
plt.show()