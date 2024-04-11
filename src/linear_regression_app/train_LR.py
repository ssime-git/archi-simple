# Import necessary libraries
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.model_selection import train_test_split  # Split data into training and test sets
import numpy as np  # Library for numerical operations
import joblib  # Library for saving and loading models
import pickle  # Library for object serialization

# Step 2: Create data for training
# Generate example data
X = 2 * np.random.rand(100, 1)  # Generate 100 random feature values
y = 4 + 3 * X + np.random.randn(100, 1)  # Generate 100 target values with some noise

# Step 3: Train the linear regression model
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the linear regression model
lin_reg = LinearRegression()  # Create a LinearRegression object
lin_reg.fit(X_train, y_train)  # Fit the model to the training data

# Step 4: Save the trained model
# Uncomment the following line to use joblib for saving the model
# joblib.dump(lin_reg, 'linear_regression_model.pkl')

# Alternatively, use pickle to serialize and save the trained model to a file
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)  # Write the model `lin_reg` to 'linear_regression_model.pkl'

# Print the generated features to the console
print(X)