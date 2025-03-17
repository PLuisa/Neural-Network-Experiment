import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the training and test data
train_data = pd.read_csv(r'C:\Uni\DadaMin\AtRiskStudentTraining.csv.csv')
test_data = pd.read_csv(r"C:\Uni\DadaMin\AtRiskStudentTest.csv")

# Define a range of architectures to explore
hidden_layers_range = [1, 2, 3 ,4 ,5, 6, 7, 8, 9, 10 ]  # Number of hidden layers
neurons_per_layer_range = [10, 20, 30]  # Number of neurons per layer

# Define the error function
def calculate_error(y_true, y_pred):
    # Use the same error function as proposed in part (b)
    return accuracy_score(y_true, y_pred)

# Initialize a dictionary to store errors for each architecture variation
errors = {}

# Iterate over each architecture variation
for hidden_layers in hidden_layers_range:
    for neurons_per_layer in neurons_per_layer_range:
        # Define and train the neural network
        model = MLPClassifier(hidden_layer_sizes=tuple(neurons_per_layer for _ in range(hidden_layers)), random_state=42)
        X_train = train_data[['GPA', 'attendance', 'duration', 'language']].values
        y_train = train_data['at-risk'].values
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        X_test = test_data[['GPA', 'attendance', 'duration', 'language']].values
        y_test = test_data['at-risk'].values
        y_pred = model.predict(X_test)
        
        # Calculate the error
        error = calculate_error(y_test, y_pred)
        
        # Store the error for this architecture variation
        errors[(hidden_layers, neurons_per_layer)] = error

# Report the errors in a table
print("Errors against Number of Hidden Layers and Neurons:")
print("Hidden Layers | Neurons per Layer | Error")
for architecture, error in errors.items():
    print("{:^13} | {:^17} | {:.4f}".format(architecture[0], architecture[1], error))

# Save the errors for reporting
with open("NN2_errors.txt", "w") as f:
    f.write("Errors against Number of Hidden Layers and Neurons:\n")
    f.write("Hidden Layers | Neurons per Layer | Error\n")
    for architecture, error in errors.items():
        f.write("{:^13} | {:^17} | {:.4f}\n".format(architecture[0], architecture[1], error))

