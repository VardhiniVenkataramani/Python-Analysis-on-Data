# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Import the Data
# Read the data from the uploaded file
data = pd.read_csv("a_batch.txt", sep="\s+", header=None, names=["Advertising", "Sales"])

# Step 3: Analyze the Data
print("Dataset Head:\n", data.head())
print("\nDataset Info:")
print(data.info())
print("\nDataset Description:\n", data.describe())

# Step 4: Declare Feature and Target Variables
X = data["Advertising"].values.reshape(-1, 1)  # Feature variable (Advertising)
y = data["Sales"].values  # Target variable (Sales)

# Step 5: Plot Scatter Plot between X and y
plt.scatter(X, y, color="blue", label="Data Points")
plt.title("Scatter Plot: Advertising vs Sales")
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Step 6: Checking and Reshaping of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 7: Apply Linear Regression Model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Print the model coefficients
print("\nLinear Regression Coefficients:")
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Step 8: Plot the Regression Line
# Plot the scatter plot again
plt.scatter(X, y, color="blue", label="Data Points")

# Generate predictions for the entire X range to plot the regression line
line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
line_y = model.predict(line_X)

# Plot the regression line
plt.plot(line_X, line_y, color="red", label="Regression Line")
plt.title("Linear Regression: Advertising vs Sales")
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Step 9: Evaluate the Model
# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error (MSE):", mse)
