import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load the test data
test_data = pd.read_csv(r"C:\Uni\DadaMin\AtRiskStudentTest.csv")

# Define the predictors
predictors = ['GPA', 'attendance', 'duration', 'language']

# Prepare the test data
X_test = test_data[predictors].values
y_test = test_data['at-risk'].values  # Assuming 'at-risk' is the target column in the test data

# Load the trained model from NN1.py
try:
    model = joblib.load(r'C:\Users\Luisa\source\repos\NN1\NN1.pkl')
except FileNotFoundError:
    print("Error: Trained model file not found.")
    exit(1)  # Exit the program if the model file is not found

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy as the error function
accuracy = accuracy_score(y_test, y_pred)

# Report the error
print("Accuracy on test data:", accuracy)

# Save your implementation in a Python script named NN1validate.py
# This code snippet can be saved in a file named "NN1validate.py"
with open("NN1validate.py", "w") as f:
    f.write("""
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load the test data
test_data = pd.read_csv("AtRiskStudentsTest.csv")

# Prepare the test data
X_test = test_data[predictors].values
y_test = test_data['at-risk'].values  

# Load the trained model from NN1.py
try:
    model = joblib.load('trained_model.pkl')
except FileNotFoundError:
    print("Error: Trained model file not found.")
    exit(1)  # Exit the program if the model file is not found

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy as the error function
accuracy = accuracy_score(y_test, y_pred)

# Report the error
print("Accuracy on test data:", accuracy)
""")
