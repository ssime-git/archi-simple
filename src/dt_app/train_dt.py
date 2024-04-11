# Step 1: Import necessary libraries
from sklearn.datasets import load_iris  # Import the Iris dataset
from sklearn.tree import DecisionTreeClassifier  # Decision Tree algorithm
from sklearn.model_selection import train_test_split  # Split data into training and test sets
import joblib  # Library for saving and loading models
import pickle  # Library for object serialization

# Step 2: Load the Iris dataset for training
iris = load_iris()  # Load the Iris dataset
X, y = iris.data, iris.target  # Separate features (X) and target labels (y)

# Step 3: Train the decision tree model
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data

# Instantiate and train the decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=2)  # Initialize the DecisionTreeClassifier with a max depth of 2
tree_clf.fit(X_train, y_train)  # Fit the model to the training data

# Step 4: Save the trained model
# Uncomment the following line to use joblib for saving the model
# joblib.dump(tree_clf, 'decision_tree_classifier.pkl')  # Save the model using joblib

# Alternatively, use pickle to serialize and save the trained model to a file
with open('decision_tree_classifier.pkl', 'wb') as f:
    pickle.dump(tree_clf, f)  # Serialize and save the model using pickle

# Print the features to the console
print(X)