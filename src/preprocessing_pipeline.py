import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import yeojohnson
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

class PreprocessingPipeline:
    def __init__(self):
        """
        Initialize preprocessing components with placeholders for saving encoders and scalers.
        """
        self.ordinal_encoder = OrdinalEncoder()
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        
        # Placeholders to store fitted transformers
        self.fitted_categorical_encoders = {}
        self.target_encoder = OrdinalEncoder()
    
    @classmethod
    def load_data(cls, path='data/clean.csv'):
        """
        Load data from the specified CSV file
        
        Args:
            path (str): Path to the CSV file
        
        Returns:
            pandas.DataFrame: Loaded dataframe
        """
        return pd.read_csv(path)
    
    def preprocess(self, df=None, save=True):
        """
        Preprocess the input dataframe
        
        Args:
            df (pandas.DataFrame, optional): Input dataframe. 
                If None, loads from data/clean.csv
            save (bool): Whether to save preprocessed data and pipeline, defaults to True
        
        Returns:
            tuple: Preprocessed features and target
        """
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/preprocessed', exist_ok=True)
        
        # Load data if not provided
        if df is None:
            df = self.load_data()
        
        # Separate features and target
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        
        # Encode target variable
        y = self.target_encoder.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                # Fit and transform categorical columns
                encoder = OrdinalEncoder()
                X[col] = encoder.fit_transform(X[col].values.reshape(-1, 1))
                
                # Store the fitted encoder
                self.fitted_categorical_encoders[col] = encoder
        
        # Apply Yeo-Johnson transformation to numeric columns (excluding already processed categorical)
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            # Add a small constant to avoid issues with zero or negative values
            X[col] = yeojohnson(X[col] + abs(X[col].min()) + 1)[0]
        
        # Scale features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Apply SMOTE for handling class imbalance
        X, y = self.smote.fit_resample(X, y)
        
        # Save preprocessed data if requested
        if save:
            self.save_preprocessed_data(X, y)
            self.save_pipeline()
        
        return X, y
    
    def save_preprocessed_data(self, X, y):
        """
        Save preprocessed features and target to CSV files
        
        Args:
            X (pandas.DataFrame): Preprocessed features
            y (numpy.ndarray): Preprocessed target
        """
        # Save preprocessed features
        X.to_csv('data/preprocessed/preprocessed_features.csv', index=False)
        
        # Save preprocessed target
        pd.Series(y).to_csv('data/preprocessed/preprocessed_target.csv', index=False)
    
    def save_pipeline(self):
        """
        Save the preprocessing pipeline components using pickle
        """
        # Save the entire pipeline
        with open('models/preprocessing_pipeline.pkl', 'wb') as f:
            pickle.dump({
                'categorical_encoders': self.fitted_categorical_encoders,
                'target_encoder': self.target_encoder,
                'scaler': self.scaler,
                'smote': self.smote
            }, f)
    
    @classmethod
    def load_pipeline(cls):
        """
        Load the saved preprocessing pipeline
        
        Returns:
            dict: Loaded pipeline components
        """
        with open('models/preprocessing_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def load_preprocessed_data(cls):
        """
        Load preprocessed data from CSV files
        
        Returns:
            tuple: Loaded preprocessed features and target
        """
        X = pd.read_csv('data/preprocessed/preprocessed_features.csv')
        y = pd.read_csv('data/preprocessed/preprocessed_target.csv')
        return X, y.values.ravel()
    
    def transform(self, df):
        """
        Transform new data using the saved preprocessing steps
        
        Args:
            df (pandas.DataFrame): Input dataframe to transform
        
        Returns:
            tuple: Transformed features and encoded target
        """
        # Create a copy of the input dataframe
        X = df.copy()
        
        # Separate target if present
        if 'Loan_Status' in X.columns:
            y = X['Loan_Status']
            X = X.drop('Loan_Status', axis=1)
            # Encode target if needed
            y = self.target_encoder.transform(y.values.reshape(-1, 1)).ravel()
        else:
            y = None
        
        # Encode categorical variables using saved encoders
        for col, encoder in self.fitted_categorical_encoders.items():
            X[col] = encoder.transform(X[col].values.reshape(-1, 1))
        
        # Apply Yeo-Johnson transformation to numeric columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            X[col] = yeojohnson(X[col] + abs(X[col].min()) + 1)[0]
        
        # Scale features
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        return (X, y) if y is not None else X

# Example of how to use the preprocessing pipeline
if __name__ == "__main__":
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline()
    
    # Preprocess data from default path (data/clean.csv)
    X, y = pipeline.preprocess()
    
    # Or preprocess from a specific dataframe
    # df = pd.read_csv('path/to/your/dataframe.csv')
    # X, y = pipeline.preprocess(df)