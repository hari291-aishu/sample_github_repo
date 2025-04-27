# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set to evaluate accuracy
y_pred = model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))

# Save the trained model using joblib or pickle
# Using pickle:
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Alternatively, using joblib:
# joblib.dump(model, 'random_forest_model.joblib')
