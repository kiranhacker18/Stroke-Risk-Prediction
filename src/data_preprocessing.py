import pandas as pd
import os

def load_data():
    # Get correct file path
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_path, 'data', 'healthcare-dataset-stroke-data.csv')
    
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    print("Before Cleaning:\n", df.isnull().sum())

    # Drop 'id' column
    df = df.drop('id', axis=1)

    # Fill missing BMI values
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    # 🔥 NEW: Create BP Category feature
    def categorize_bp(row):
        if row['hypertension'] == 1:
            return 'high'
        else:
            return 'normal'   # dataset has no low BP info

    df['bp_category'] = df.apply(categorize_bp, axis=1)

    # 🔥 Remove old hypertension column
    df = df.drop('hypertension', axis=1)

    # Convert categorical variables to numeric
    df = pd.get_dummies(df, drop_first=True)

    print("\nAfter Cleaning:\n", df.isnull().sum())

    return df

