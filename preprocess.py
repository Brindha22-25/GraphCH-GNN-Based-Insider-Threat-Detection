import pandas as pd
import numpy as np
from datetime import datetime

# Load dataset
input_file = "../data/raw/sampled_dataset.csv"
output_file = "../data/processed/processed_data.csv"

def preprocess_data(input_file, output_file):
    #df = pd.read_csv(input_file)
    df = pd.read_csv(input_file, dtype={'employee_name': str, 'user_id': str}, low_memory=False)


    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract useful time features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_after_hours'] = df['hour'].apply(lambda x: 1 if x < 9 or x > 18 else 0)

    # Count number of recipients
    df['num_recipients'] = df[['to', 'cc', 'bcc']].apply(lambda x: sum(pd.notna(x)), axis=1)

    # Flag external emails
    df['external_email'] = df['to'].apply(lambda x: 1 if isinstance(x, str) and '@dtaa.com' not in x else 0)

    # Replace missing values
    df.fillna({'attachments': 0, 'size': df['size'].median()}, inplace=True)

    # Drop unnecessary columns
    df.drop(columns=['id', 'employee_name', 'user_id'], inplace=True, errors='ignore')

    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(input_file, output_file)
