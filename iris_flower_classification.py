"""
Iris Flower Classification - Machine Learning Model
Author: SamiUllah Saif - ML Engineer
Goal: Classify iris species based on sepal and petal measurements
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION MODEL")
print("Author: SamiUllah Saif - ML Engineer")
print("=" * 60)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

print(f"\n📊 Dataset Information:")
print(f"   - Total samples: {X.shape[0]}")
print(f"   - Features: {', '.join(iris.feature_names)}")
print(f"   - Target classes: {', '.join(iris.target_names)}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Performance:")
print(f"   - Algorithm: Random Forest Classifier")
print(f"   - Accuracy: {accuracy:.2%}")
print(f"   - Test size: 20% ({X_test.shape[0]} samples)")

print(f"\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
print(f"\n🔍 Feature Importance:")
for feature, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"   - {feature}: {importance:.3f}")

print("\n" + "=" * 60)
print("✅ Model training completed successfully!")
print("=" * 60)
