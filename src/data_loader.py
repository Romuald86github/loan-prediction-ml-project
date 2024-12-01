# Data Loading Module
import os
import pandas as pd
import config

def load_data(file_path=None):
    """
    Load loan prediction dataset, clean it, and save to a CSV file
    
    Args:
        file_path (str, optional): Path to the dataset. Defaults to None.
    
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Use default source if no path provided
    if file_path is None:
        file_path = config.DATA_SOURCE
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Columns to be dropped (based on original notebook)
    columns_to_drop = [
        'Loan_ID', 
        'CoapplicantIncome', 
        'ApplicantIncome', 
        'Self_Employed', 
        'Loan_Amount_Term'
    ]
    
    # Drop specified columns
    df = df.drop(columns=columns_to_drop, axis=1)
    
    # Handle missing values
    # Numeric columns
    if df['LoanAmount'].isnull().any():
        df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['LoanAmount'] = df['LoanAmount'].astype(int)
    
    # Categorical columns
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Credit_History', 'Property_Area']
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Save cleaned data to CSV
    output_path = 'data/clean.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    # If script is run directly, load and clean the data
    load_data()