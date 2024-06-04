import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Load the data
data = fetch_olivetti_faces()
faces = data.images
n_samples, h, w = faces.shape

# Flatten images
X = faces.reshape(n_samples, h * w)

# PCA
n_components = 100  # Number of components for illustration
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
X_transformed = pca.fit_transform(X)
eigenfaces = pca.components_.reshape((n_components, h, w))

# Reconstruct faces with different number of components
components = [5, 10, 30, 60, 100]
explained_variances = np.cumsum(pca.explained_variance_ratio_)
fig, axes = plt.subplots(2, len(components) + 1, figsize=(15, 8))

# Original image
axes[0, 0].imshow(faces[0], cmap=plt.cm.gray)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Reconstructed images
for i, n in enumerate(components):
    # Proper reconstruction using dot product
    X_reconstructed = np.dot(X_transformed[0, :n], eigenfaces[:n].reshape(n, h*w)) + pca.mean_
    X_reconstructed_image = X_reconstructed.reshape(h, w)
    axes[0, i+1].imshow(X_reconstructed_image, cmap=plt.cm.gray)
    axes[0, i+1].set_title(f'{n} Components')
    axes[0, i+1].axis('off')
    
    # Show variance explained
    axes[1, i].bar(range(n), explained_variances[:n], alpha=0.5)
    axes[1, i].set_title(f'Variance: {explained_variances[n-1]:.2f}')
    axes[1, i].set_ylim([0, 1])

axes[1, len(components)].plot(np.arange(1, 101), explained_variances, marker='o')
axes[1, len(components)].set_title('Cumulative Variance')
axes[1, len(components)].set_xlabel('Number of Components')
axes[1, len(components)].set_ylabel('Variance Explained')

plt.tight_layout()
plt.show()
