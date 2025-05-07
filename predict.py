import pandas as pd
import joblib

# Load trained model
MODEL_PATH = "../models/insider_threat_model.pkl"
model = joblib.load(MODEL_PATH)

# Load feature datasets
bart_df = pd.read_csv("../data/features/bart_features.csv")
delay_df = pd.read_csv("../data/features/delay_discounting.csv")

# Merge datasets on "user" column
df = pd.merge(bart_df, delay_df, on="user")

# Ensure correct features
required_features = ["bart_risk_score", "delay_discounting_score"]
if not all(feature in df.columns for feature in required_features):
    raise ValueError(f"Missing required features: {required_features}")

# Make predictions
df["insider_threat"] = model.predict(df[["bart_risk_score", "delay_discounting_score"]])

# Identify high-risk users
threats = df[df["insider_threat"] == 1]["user"].tolist()

# Check if system is unsafe
if len(threats) > 0:
    system_status = "⚠️ SYSTEM IS UNSAFE!"
    print(f"{system_status} High-Risk Users: {', '.join(threats)}")
else:
    system_status = "✅ SYSTEM IS SAFE!"
    print(system_status)

# Save predictions
#df.to_csv("../data/predictions.csv", index=False)
#print("✅ Predictions saved to ../data/predictions.csv")
# Print predictions for debugging
print(df[["user", "bart_risk_score", "delay_discounting_score", "insider_threat"]])

