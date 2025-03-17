#Code based on https://www.pluralsight.com/resources/blog/guides/machine-learning-neural-networks-scikit-learn

# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

# Enable interactive mode
plt.ion()

# Load the training data from the CSV file
train_data = pd.read_csv(r"C:\Uni\DadaMin\AtRiskStudentTraining.csv.csv")


#Preparing the trainind data

predictors = ['GPA', 'attendance', 'duration', 'language']
target_column = 'at-risk'


# Prepare the training data
X_train = train_data[predictors].values
y_train = train_data['at-risk'].values
    

model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', random_state=42)

model.fit(X_train, y_train)

print("Model parameters after training:")
print(model)
print(X_train.shape); print(y_train.shape)