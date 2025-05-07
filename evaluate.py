import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test dataset
bart_df = pd.read_csv("../data/features/bart_features.csv")
delay_df = pd.read_csv("../data/features/delay_discounting.csv")

# Merge features
df = bart_df.merge(delay_df, on="user")

# Load trained model
model = joblib.load("../models/insider_threat_model.pkl")

# Define labels (same method used in train.py)
df['insider_threat'] = ((df['bart_risk_score'] > 0.6) & (df['delay_discounting_score'] > 0.6)).astype(int)

# Prepare data
X = df[['bart_risk_score', 'delay_discounting_score']]
y_true = df['insider_threat']

# Make predictions
y_pred = model.predict(X)

# Evaluate model
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Model Evaluation Results:")
print(f" Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")