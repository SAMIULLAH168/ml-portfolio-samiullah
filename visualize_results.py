import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Generating visualizations...")

# Diabetes Model
data_diabetes = load_diabetes()
X_diabetes, y_diabetes = data_diabetes.data, data_diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

model_diabetes = RandomForestRegressor(n_estimators=100, random_state=42)
model_diabetes.fit(X_train, y_train)

# Plot feature importance
importance = model_diabetes.feature_importances_
features = data_diabetes.feature_names

plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='skyblue')
plt.xlabel('Importance Score')
plt.title('Diabetes Prediction - Feature Importance')
plt.tight_layout()
plt.savefig('diabetes_feature_importance.png')
print("✓ Saved: diabetes_feature_importance.png")

# Iris Model
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

model_iris = RandomForestClassifier(n_estimators=100, random_state=42)
model_iris.fit(X_train, y_train)

importance_iris = model_iris.feature_importances_
features_iris = iris.feature_names

plt.figure(figsize=(10, 6))
plt.barh(features_iris, importance_iris, color='lightcoral')
plt.xlabel('Importance Score')
plt.title('Iris Classification - Feature Importance')
plt.tight_layout()
plt.savefig('iris_feature_importance.png')
print("✓ Saved: iris_feature_importance.png")

print("✅ Visualizations complete!")
