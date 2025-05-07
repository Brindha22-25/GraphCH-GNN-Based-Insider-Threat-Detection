import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # For oversampling the minority class
import joblib

# Ensure models directory exists
os.makedirs("../models", exist_ok=True)

# Load feature datasets
bart_df = pd.read_csv("../data/features/bart_features.csv")
delay_df = pd.read_csv("../data/features/delay_discounting.csv")

# Merge features
df = bart_df.merge(delay_df, on="user")

# Debug: Check original dataset shape
print(f"Original dataset shape: {df.shape}")

# Sample a fraction of the data to reduce memory usage
df = df.sample(frac=0.001, random_state=42)  # Adjust fraction as needed
print(f"Sampled dataset shape: {df.shape}")

# Define labels: insider threat if both scores exceed threshold or any of them exceed 0.5 (unsafe system)
df['insider_threat'] = (
    ((df['bart_risk_score'] > 0.7) & (df['delay_discounting_score'] > 0.7)) | 
    ((df['bart_risk_score'] > 0.5) & (df['delay_discounting_score'] > 0.5))
).astype(int)

# Prepare data
X = df[['bart_risk_score', 'delay_discounting_score']]
y = df['insider_threat']

# Debugging: Check types and label distribution
print("Feature types:")
print(X.dtypes)
print("Label distribution:")
print(y.value_counts())

# Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Debugging: Check balanced class distribution after SMOTE
print("Resampled label distribution:")
print(pd.Series(y_res).value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model with fewer trees and limited parallelism to reduce memory load
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2f}")

# Print classification report (precision, recall, f1-score)
print("🎯 Classification Report:\n", classification_report(y_test, y_pred))

# Save trained model
model_path = "../models/insider_threat_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved at {model_path}")
