#!/usr/bin/env python3
"""
Run all ML portfolio projects
Author: SamiUllah Saif - ML Engineer
"""

print("\n" + "="*70)
print(" " * 20 + "ML PORTFOLIO - SAMIULLAH SAIF")
print("="*70)

# Run Iris Classification
print("\n📊 PROJECT 1: Iris Flower Classification")
print("-"*50)
exec(open('iris_flower_classification.py').read())

# Run Diabetes Prediction
print("\n📊 PROJECT 2: Diabetes Progression Prediction")
print("-"*50)
exec(open('diabetes_ml.py').read())

# Run Visualizations (if matplotlib is available)
try:
    import matplotlib
    print("\n📊 PROJECT 3: Model Visualizations")
    print("-"*50)
    exec(open('visualize_results.py').read())
except ImportError:
    print("\n⚠️  Matplotlib not installed. Run: pip install matplotlib")
except FileNotFoundError:
    print("\n⚠️  visualize_results.py not found. Run the visualization setup script.")

print("\n" + "="*70)
print("✅ ALL PROJECTS COMPLETED SUCCESSFULLY!")
print("="*70)
