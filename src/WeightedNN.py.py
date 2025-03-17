import numpy as np


# Function to calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    # Initialize distance to 0
    distance = 0.0
    # Loop over each dimension of the vectors
    for i in range(len(row1)):
        # Add the squared difference of each dimension to the distance
        distance += (row1[i] - row2[i])**2
    # Return the square root of the sum of squared differences
    return np.sqrt(distance)

def make_weights(train_data, new_data):
    # Calculate the Euclidean distances between each training point and the new point
    distances = [euclidean_distance(row, new_data) for row in train_data]
    # Calculate the sum of the distances
    sum_distances =np.sum(distances)
    # Calculate weights based on the distances, where weights are proportional to distances
    weights = [distance/sum_distances for distance in distances]
    return weights

def weighted_nn(train_data, train_labels, new_data):
    # Calculate the Euclidean distances between the training points and the new point
    distances = [euclidean_distance(row, new_data) for row in train_data]

    # Calculate weights 
    weights = np.array(make_weights(train_data, new_data))

    # Convert train_labels to NumPy array
    train_labels = np.array(train_labels)

    # Calculate the weighted vote
    weighted_votes = np.sum(weights * train_labels) / np.sum(weights)

    # Round the weighted vote to get the predicted class label
    return round(weighted_votes)

# Example usage:
if __name__ == "__main__":
    # Example training data and labels
    train_data = np.array([
        [0.32, 0.43, 0.54, 0.65, 0.76], 
        [0.26, 0.54, 0.32, 0.67, 0.89], 
        [0.27, 0.60, 0.78, 0.45, 0.23], 
        [0.37, 0.36, 0.59, 0.75, 0.34], 
        [0.37, 0.68, 0.39, 0.88, 0.49]
    ])                          
    train_labels = np.array([0, 0, 0, 0, 1])
    # New data point to classify
    new_data = np.array([0.62, 0.35, 0.48, 0.76, 0.24])
    # Perform weighted nearest neighbor classification
    prediction = weighted_nn(train_data, train_labels, new_data)
    # Print the predicted class label
    print("Predicted class label:", prediction)
