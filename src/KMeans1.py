import numpy as np
import pandas as pd

def initialize_centroids(k, data):
    """
    Initializes centroids randomly.
    Returns a numpy array with centroids.
    """
    np.random.seed(42)
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = np.array([data[i] for i in centroids_indices])
    return centroids

def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def assign_centroid(data, centroids):
    """
    Assigns each observation to the closest centroid.
    Returns a list of centroid assignments for each observation.
    """
    centroid_assignments = []
    errors = []

    for observation in data:
        distances = [euclidean_distance(observation, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        centroid_error = distances[closest_centroid_index]
        centroid_assignments.append(closest_centroid_index)
        errors.append(centroid_error)

    return centroid_assignments, errors

def calculate_new_centroids(data, centroid_assignments, k):
    """
    Calculates new centroids based on current centroid assignments.
    Returns a numpy array with the new centroids.
    """
    new_centroids = []
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if centroid_assignments[j] == i]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def k_means(data, k, max_iter=300):
    """
    Executes the K-Means algorithm.
    Returns a list of centroid assignments for each observation and the final centroids.
    """
    centroids = initialize_centroids(k, data)
    for _ in range(max_iter):
        centroid_assignments, _ = assign_centroid(data, centroids)
        new_centroids = calculate_new_centroids(data, centroid_assignments, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroid_assignments, centroids

def main1():
    # Import data
    file_path = r'C:\Uni\DadaMin\HouseHoldWealth.csv'  
    df = pd.read_csv(file_path, dtype={'household_total_assets': float, 'annual_household_income': float})

    # Convert DataFrame to numpy array
    data = df[['household_total_assets', 'annual_household_income']].values

    # Define the number of clusters
    k = 3

    # Execute K-Means algorithm
    centroid_assignments, centroids = k_means(data, k)

    # Print the final centroids
    print("Final Centroids:")
    for centroid in centroids:
        print(centroid)

if __name__ == "__main__":
    main1()


