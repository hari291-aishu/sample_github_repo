# train_and_save_model.py

import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# 5. Save the model using pickle
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'iris_model.pkl'")
