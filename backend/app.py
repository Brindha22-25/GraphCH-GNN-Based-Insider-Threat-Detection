import os
import streamlit as st
import pandas as pd
import joblib

# Load trained model
MODEL_PATH = r"D:\insider_threat_detection\backend\models\insider_threat_model.pkl"  # Ensure correct path format

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.sidebar.success(f"✅ Model loaded successfully!")
else:
    model = None
    st.sidebar.error("⚠️ Model file not found! Train the model first.")

# Streamlit UI
st.title("🔍 Insider Threat Detection")
st.write("Analyze user activity risk scores to detect potential insider threats.")

# File Upload Option (for batch processing)
uploaded_file = st.file_uploader("📂 Upload a CSV file with user activity data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure correct features are present
    required_features = ["user", "bart_risk_score", "delay_discounting_score"]
    if not all(feature in df.columns for feature in required_features):
        st.error(f"⚠️ Missing required features: {required_features}")
    else:
        # Make predictions
        df["insider_threat"] = model.predict(df[["bart_risk_score", "delay_discounting_score"]])

        # Identify insider threats
        threats = df[df["insider_threat"] == 1]["user"].tolist()

        # Display system status
        if len(threats) > 0:
            st.error(f"⚠️ Insider Threats Detected! System is UNSAFE! High-Risk Users: {', '.join(threats)}")
        else:
            st.success("✅ No insider threats detected. System is SAFE!")

        # Display results
        st.write("### Prediction Results:")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions", data=csv, file_name="insider_threat_predictions.csv", mime="text/csv")

# Manual Input for a Single User
st.subheader("🔹 Manually Enter Data for Prediction")
user_id = st.text_input("User ID", "")
bart_risk_score = st.number_input("BART Risk Score", min_value=0.0, max_value=1.0, step=0.01)
delay_discounting_score = st.number_input("Delay Discounting Score", min_value=0.0, max_value=1.0, step=0.01)

if st.button("🔍 Predict Insider Threat"):
    if model:
        single_input = pd.DataFrame([[bart_risk_score, delay_discounting_score]], 
                                    columns=["bart_risk_score", "delay_discounting_score"])
        prediction = model.predict(single_input)[0]
        if prediction == 1:
            st.error(f"⚠️ Insider Threat Detected for {user_id}! System is at RISK!")
        else:
            st.success(f"✅ {user_id} is Safe. No Insider Threat Detected.")
    else:
        st.error("⚠️ Model not found! Train the model first.")
