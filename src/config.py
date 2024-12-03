# Configuration Module

# Data source configuration
DATA_SOURCE = 'https://github.com/dsrscientist/DSData/raw/master/loan_prediction.csv'

# Model configuration
MODEL_PARAMS = {
    'n_estimators': 50,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'random_state': 42
}

# Train-test split configuration
TEST_SIZE = 0.25
RANDOM_STATE = 42