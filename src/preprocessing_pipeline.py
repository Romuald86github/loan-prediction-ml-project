import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class PreprocessingPipeline:
    def __init__(self):
        self.preprocessor = None
        self.smote = SMOTE(random_state=42)
        self.feature_names = None
        self.target_encoder = LabelEncoder()

    @classmethod
    def load_data(cls, path='data/clean.csv'):
        return pd.read_csv(path)

    def preprocess(self, df=None, save=True):
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/preprocessed', exist_ok=True)
        
        if df is None:
            df = self.load_data()
        
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        # Encode target variable
        y = self.target_encoder.fit_transform(y)

        numeric_features = ['LoanAmount']
        categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Credit_History', 'Property_Area']

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_preprocessed = self.preprocessor.fit_transform(X)
        
        onehot_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names(categorical_features).tolist()
        self.feature_names = numeric_features + cat_feature_names

        X_resampled, y_resampled = self.smote.fit_resample(X_preprocessed, y)
        
        if save:
            self.save_preprocessed_data(X_resampled, y_resampled)
            self.save_pipeline()
        
        return X_resampled, y_resampled

    def save_preprocessed_data(self, X, y):
        np.savez('data/preprocessed/preprocessed_data.npz', X=X, y=y)

    def save_pipeline(self):
        with open('models/preprocessing_pipeline.pkl', 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'smote': self.smote,
                'feature_names': self.feature_names,
                'target_encoder': self.target_encoder
            }, f)

    @classmethod
    def load_pipeline(cls):
        with open('models/preprocessing_pipeline.pkl', 'rb') as f:
            pipeline_components = pickle.load(f)
        
        pipeline = cls()
        pipeline.preprocessor = pipeline_components['preprocessor']
        pipeline.smote = pipeline_components['smote']
        pipeline.feature_names = pipeline_components['feature_names']
        pipeline.target_encoder = pipeline_components['target_encoder']
        
        return pipeline

    @classmethod
    def load_preprocessed_data(cls):
        data = np.load('data/preprocessed/preprocessed_data.npz')
        return data['X'], data['y']

    def transform(self, df):
        return self.preprocessor.transform(df)

if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    X, y = pipeline.preprocess()