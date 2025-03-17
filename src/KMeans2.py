#Code based on https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
#https://mayankdw.medium.com/k-means-clustering-and-dunn-index-implementaion-from-scratch-9c66573bfe90 

import pandas as pd
import numpy as np

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
    Returns a list of centroid assignments for each observation, the final centroids, and the inertia.
    """
    centroids = initialize_centroids(k, data)
    inertia = 0
    for _ in range(max_iter):
        centroid_assignments, errors = assign_centroid(data, centroids)
        new_centroids = calculate_new_centroids(data, centroid_assignments, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
        inertia = sum(errors)
    return centroid_assignments, centroids, inertia

def dunn_index(centroids, centroid_assignments, data):
    """
    Calculates the Dunn index for a given clustering.
    Returns the Dunn index value.
    """
    intra_cluster_distances = []
    for centroid_index, centroid in enumerate(centroids):
        cluster_points = np.array([data[i] for i in range(len(data)) if centroid_assignments[i] == centroid_index])
        centroid_distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_cluster_distances.extend(centroid_distances)

    min_inter_cluster_distance = np.min([np.min([euclidean_distance(centroids[i], centroids[j]) for j in range(len(centroids)) if i != j]) for i in range(len(centroids))])
    dunn_index = min_inter_cluster_distance / np.max(intra_cluster_distances)
    return dunn_index

def main():
    # Importing the data
    file_path = r"C:\Uni\DadaMin\HouseHoldWealth.csv"
    df = pd.read_csv(file_path, dtype={'household_total_assets': float, 'annual_household_income': float})
    
    # Convert DataFrame to numpy array
    data = df.values

    # Calculate Dunn index and inertia for each K value from 2 to 10
    dunn_indices = []
    inertias = []

    for k in range(2, 11):
        centroid_assignments, centroids, inertia = k_means(data, k)
        dunn = dunn_index(centroids, centroid_assignments, data)
        dunn_indices.append(dunn)
        inertias.append(inertia)
        
        # Print results
        print(f"K={k}:")
        print("Centroids:")
        print(centroids)
        print(f"Dunn Index: {dunn}")
        print(f"Inertia: {inertia}")
        print()

    # Print or save the Dunn index and inertia values along with corresponding K values
    print("K | Dunn Index | Inertia")
    for k, dunn, inertia in zip(range(2, 11), dunn_indices, inertias):
        print(f"{k} | {dunn} | {inertia}")

if __name__ == "__main__":
    main()
