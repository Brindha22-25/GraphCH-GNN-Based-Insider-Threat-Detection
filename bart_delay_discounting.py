import pandas as pd
import numpy as np

# Load processed dataset
input_file = "../data/processed/processed_data.csv"
bart_output_file = "../data/features/bart_features.csv"
delay_discounting_output_file = "../data/features/delay_discounting.csv"

def compute_bart_features(df):
    # Simulated BART risk score: High risk = sends many large emails to external contacts
    df['bart_risk_score'] = df['external_email'] * df['size'] * df['num_recipients']
    df['bart_risk_score'] = (df['bart_risk_score'] - df['bart_risk_score'].min()) / (df['bart_risk_score'].max() - df['bart_risk_score'].min())  # Normalize
    return df[['user', 'bart_risk_score']]

def compute_delay_discounting(df):
    # Delay Discounting Score: Measures impulsive behavior (high after-hour emails)
    df['delay_discounting_score'] = df['is_after_hours'] * df['num_recipients']
    df['delay_discounting_score'] = (df['delay_discounting_score'] - df['delay_discounting_score'].min()) / (df['delay_discounting_score'].max() - df['delay_discounting_score'].min())  # Normalize
    return df[['user', 'delay_discounting_score']]

if __name__ == "__main__":
    df = pd.read_csv(input_file)
    
    # bart_features = compute_bart_features(df)
    # bart_features.to_csv(bart_output_file, index=False)
    # print(f"BART Features saved to {bart_output_file}")

    # delay_discounting_features = compute_delay_discounting(df)
    # delay_discounting_features.to_csv(delay_discounting_output_file, index=False)
    # print(f"Delay Discounting Features saved to {delay_discounting_output_file}")
#print(df[['user', 'external_email', 'size', 'num_recipients', 'bart_risk_score']].head(10))
print("Available columns:", df.columns)

