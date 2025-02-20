import numpy as np
np.random.seed(1)
from scipy.spatial.distance import cdist
# Example: 10 true models (each with 5 parameters)
true_models = np.random.rand(10, 5)

# Example: 12 estimated models (each with 5 parameters)
estimated_models = np.random.rand(12, 5)

# Compute Euclidean distances manually using NumPy broadcasting
# (true_models[:, None, :] creates a shape (10, 1, 5) to allow broadcasting)
distances = np.linalg.norm(true_models[:, None, :] - estimated_models[None, :, :], axis=2)
#distances = cdist(true_models, estimated_models, metric='euclidean')
# Find the closest estimated model for each true model
min_distances = np.min(distances, axis=1)  # Minimum distance for each true model
closest_estimated_indices = np.argmin(distances, axis=1)  # Index of closest estimated model

# Print results
for true_idx, (est_idx, distance) in enumerate(zip(closest_estimated_indices, min_distances)):
    print(f"True model {true_idx} is closest to Estimated model {est_idx} (Distance: {distance:.4f})")

# Compute overall average minimal distance
avg_min_distance = np.mean(min_distances)
print(f"\nAverage minimum distance across all true models: {avg_min_distance:.4f}")
