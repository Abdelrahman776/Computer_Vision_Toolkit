import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Step 3: Perform PCA
pca = PCA()
pca.fit(centered_faces)

# Explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

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
plt.ylabel('Cumulative Variance')
plt.title('Variance (Original)')

# Reconstructions
for i, k in enumerate(k_values):
    reconstructed = np.dot(pca.transform(centered_faces)[:, :k], pca.components_[:k, :]) + mean_face
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
    plt.ylabel('Variance')
    plt.title(f'Variance: {explained_variance[k - 1]:.2f}')

plt.tight_layout()
plt.show()

#############################################

# Define the path to the data

# Load the random image
random_image_path = 'subject02.noglasses.jpg'
random_image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
if random_image is None:
    print("Error: Could not read the random image.")
    exit()

# Resize the random image to match the size used in the dataset
random_image = cv2.resize(random_image, (50, 50))

# Flatten the random image
random_face = random_image.flatten()

# Project the random face onto the PCA space
random_face_centered = random_face - mean_face
random_face_projected = pca.transform(random_face_centered.reshape(1, -1))

# Calculate Euclidean distances between the random face and each face in the dataset
distances = np.linalg.norm(pca.transform(centered_faces) - random_face_projected, axis=1)

# Define a threshold
threshold = 4000  # Adjust this threshold based on your dataset

# Find the index of the closest match
closest_index = np.argmin(distances)
closest_distance = distances[closest_index]

if closest_distance < threshold:
    closest_label = labels[closest_index]
    print(f"The random image looks like {closest_label} from the dataset.")
    
    # Plot the random image and the closest match
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(random_image, cmap='gray')
    axes[0].set_title('Random Image')
    axes[0].axis('off')
    
    closest_image = faces[closest_index].reshape(50, 50)
    axes[1].imshow(closest_image, cmap='gray')
    axes[1].set_title(f'Closest Match: {closest_label}')
    axes[1].axis('off')
    
    plt.show()
else:
    print("The random image doesn't match anyone from the dataset.")