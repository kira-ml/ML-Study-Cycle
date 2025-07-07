import numpy as np
import random

# In this implementation, I demonstrate how to build a simple k-means clustering algorithm from scratch.
# The algorithm iteratively assigns data points to the nearest centroid and updates those centroids until convergence.
# This implementation serves as a hands-on introduction to unsupervised learning and clustering mechanics.


# Calculates the Euclidean distance between two points.
# Euclidean distance is the most common metric for similarity in continuous feature space.
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# Randomly selects `k` data points from the dataset to serve as initial centroids.
# Using `random.sample` ensures the centroids are unique and selected from actual data.
def initialize_centroids(data, k):
    indices = random.sample(range(len(data)), k)
    return np.array([data[i] for i in indices])


# Assigns each data point to the cluster of its closest centroid.
# This step partitions the data by calculating distances from each point to all centroids.
def assign_cluster(data, centroids):
    cluster = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest = np.argmin(distances)  # Index of the nearest centroid
        cluster.append(closest)
    return np.array(cluster)


# Updates centroid positions based on the mean of points assigned to each cluster.
# If a cluster has no assigned points, we reinitialize its centroid randomly to avoid collapse.
def update_centroids(data, cluster, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[cluster == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)  # Re-center the cluster
        else:
            new_centroid = data[random.randint(0, len(data)-1)]  # Random recovery for empty cluster
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


# Checks if centroids have stabilized by measuring the total movement since the last iteration.
# Convergence is determined when the shift in centroid positions is below a small threshold.
def has_converged(old_centroids, new_centroids, threshold=1e-4):
    distance = [euclidean_distance(old, new) for old, new in zip(old_centroids, new_centroids)]
    return sum(distance) < threshold


# The main routine that drives the k-means algorithm.
# It initializes centroids, iteratively updates them, and terminates once centroids converge or max iterations is reached.
def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)

    for _ in range(max_iterations):
        clusters = assign_cluster(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# Set seeds for reproducibility across NumPy and Python’s random module.
random.seed(42)
np.random.seed(42)

# Generate synthetic data: two Gaussian clusters in 2D space.
# Each cluster is centered at a different location with standard deviation of 1.
data = np.vstack([
    np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
    np.random.normal(loc=[5, 5], scale=1, size=(100, 2))
])

# Set the number of clusters.
k = 2

# Run the k-means clustering algorithm on the synthetic dataset.
clusters, centroids = k_means(data, k)

# Output the results: final centroid positions and first 10 cluster assignments.
print("Final centroids:")
print(centroids)
print("\nCluster assignment for first 10 points:")
print(clusters[:10])
