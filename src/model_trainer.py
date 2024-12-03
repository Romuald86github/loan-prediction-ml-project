import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing_pipeline import PreprocessingPipeline

def train_and_evaluate(X, y, model, model_name):
    """
    Train and evaluate a machine learning model with cross-validation.
    
    Args:
        X (array-like): Input features
        y (array-like): Target labels
        model (estimator): Machine learning model to train
        model_name (str): Name of the model for logging
    
    Returns:
        dict: Evaluation metrics and model details
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    return {
        'model_name': model_name,
        'cross_val_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'trained_model': model
    }

def hyperparameter_tuning(X, y, model_class, param_grid):
    """
    Perform hyperparameter tuning for a given model
    
    Args:
        X (array-like): Input features
        y (array-like): Target labels
        model_class (estimator): Machine learning model class
        param_grid (dict): Hyperparameter grid for search
    
    Returns:
        dict: Best model and its performance metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        model_class(), 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model and predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

def main():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create preprocessing pipeline
    preprocessing = PreprocessingPipeline()
    
    # Load and preprocess data
    X, y = preprocessing.preprocess()
    
    # Model configuration
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Model parameters
    model_params = {
        'Random Forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression,
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            }
        },
        'SVM': {
            'model': SVC,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    # Tune and evaluate models
    tuned_models = {}
    for name, config in model_params.items():
        print(f"Tuning {name}...")
        tuned_models[name] = hyperparameter_tuning(X, y, config['model'], config['params'])
        print(f"{name} Best Params: {tuned_models[name]['best_params']}")
        print(f"{name} Accuracy: {tuned_models[name]['accuracy']:.4f}\n")
    
    # Select best model
    best_model_name = max(tuned_models, key=lambda x: tuned_models[x]['accuracy'])
    best_model = tuned_models[best_model_name]['model']
    
    print(f"Best Model: {best_model_name}")
    print("Classification Report:")
    print(tuned_models[best_model_name]['classification_report'])
    
    # Save best model
    model_filename = f'models/{best_model_name.replace(" ", "_")}_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Best model saved: {model_filename}")

if __name__ == "__main__":
    main()