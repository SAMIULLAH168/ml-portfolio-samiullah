# Diabetes Prediction ML Model - SamiUllah Saif
# This predicts if someone has diabetes based on medical data

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

print("="*50)
print("ML MODEL: Diabetes Progression Predictor")
print("Author: SamiUllah Saif - ML Engineer")
print("="*50)

# Load real medical data
data = load_diabetes()
X = data.data  # Medical measurements
y = data.target  # Disease progression score

print(f"\n📊 Dataset: {X.shape[0]} patients, {X.shape[1]} medical features")
print(f"🎯 Target: Disease progression score (0-300)")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
score = r2_score(y_test, predictions)

print(f"\n✅ Model: Random Forest Regressor")
print(f"✅ R² Score: {score:.2%}")
print(f"\n💡 This model explains {score:.2%} of disease progression")
print("="*50)
