import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and validate the dataset."""
    data_dir = "dataset"
    required_files = ['train.csv', 'val.csv', 'test.csv']
    
    # Load datasets
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    return train, val, test

def prepare_data(train, val, test):
    """Prepare features and target variables with enhanced feature engineering."""
    # Basic features
    features = [
        'temperature', 'humidity', 'rainfall', 
        'sanitation_score', 'population_density',
        'month', 'week'
    ]
    
    target = "risk_score"
    
    # Feature engineering function
    def engineer_features(df):
        # Cyclic encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Interaction terms
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['rainfall_pop_density'] = df['rainfall'] * (df['population_density'] / 1000)
        
        # Rolling statistics (if you have temporal data)
        # This assumes your data is sorted by time
        for window in [4, 8, 12]:  # 4, 8, and 12 week windows
            df[f'rolling_avg_temp_{window}w'] = df.groupby('zone')['temperature'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        return df
    
    # Apply feature engineering
    train = engineer_features(train)
    val = engineer_features(val)
    test = engineer_features(test)
    
    # Update features list with new features
    features = [
        'temperature', 'humidity', 'rainfall', 
        'sanitation_score', 'population_density',
        'month_sin', 'month_cos',
        'temp_humidity', 'rainfall_pop_density',
        'rolling_avg_temp_4w', 'rolling_avg_temp_8w', 'rolling_avg_temp_12w'
    ]
    
    # Prepare features and targets
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, features, target

def train_best_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with hyperparameter tuning."""

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create the model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )

    # Use GridSearchCV for hyperparameter tuning (no early stopping here)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    # Fit GridSearchCV
    print("\nPerforming hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(f"\nBest parameters: {best_params}")

    # Train the final model with early stopping using the validation set
    best_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )

    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        
        verbose=100
    )

    return best_model


def evaluate_model(model, X, y, set_name="Set"):
    """Evaluate model performance with enhanced metrics."""
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'accuracy_5pct': (np.abs(y - y_pred) <= 5).mean() * 100,
        'accuracy_10pct': (np.abs(y - y_pred) <= 10).mean() * 100,
        'accuracy_15pct': (np.abs(y - y_pred) <= 15).mean() * 100
    }
    
    # Print metrics
    print(f"\n{set_name} Metrics:")
    print("-" * 70)
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Accuracy (within 5 points): {metrics['accuracy_5pct']:.2f}%")
    print(f"Accuracy (within 10 points): {metrics['accuracy_10pct']:.2f}%")
    print(f"Accuracy (within 15 points): {metrics['accuracy_15pct']:.2f}%")
    print("-" * 70)
    
    return metrics

def main():
    print("="*70)
    print("ENHANCED DENGUE RISK PREDICTION MODEL TRAINING".center(70))
    print("="*70)
    
    try:
        # Load and prepare data
        print("\nLoading and preparing data...")
        train, val, test = load_data()
        X_train, X_val, X_test, y_train, y_val, y_test, features, target = prepare_data(train, val, test)
        
        # Combine train and validation sets
        X_full_train = pd.concat([X_train, X_val])
        y_full_train = pd.concat([y_train, y_val])
        
        # Train the best model
        print("\nTraining final model with hyperparameter tuning...")
        model = train_best_model(X_full_train, y_full_train, X_val, y_val)
        
        # Evaluate on test set
        print("\n" + "="*70)
        print("FINAL MODEL EVALUATION ON TEST SET".center(70))
        print("="*70)
        test_metrics = evaluate_model(model, X_test, y_test, "Test Set")
        
        # Save model and predictions
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/xgboost_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        print(f"\n✅ Model saved to {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    print("\n✅ Training completed successfully!")
    return 0

if __name__ == "__main__":
    main()