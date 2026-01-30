import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """
    Load the dataset and handle initial date parsing.
    """
    # Load data and combine date columns into a single index
    df = pd.read_csv(filepath, parse_dates={'date': ['year', 'month', 'day', 'hour']}, index_col='date')
    return df

def preprocess_data(df):
    """
    Clean the data: handle missing values and encode categorical variables.
    """
    # Filling missing pollution values with 0 (common for this dataset's first rows)
    df['pollution'] = df['pollution'].fillna(0)
    
    # Simple encoding: Convert wind direction (wnd_dir) to dummy variables
    df = pd.get_dummies(df, columns=['wnd_dir'])
    
    # Drop the 'No' column if it exists as it is just a row identifier
    if 'No' in df.columns:
        df = df.drop(columns=['No'])
    
    return df

def train_baseline():
    """
    Main training pipeline for the baseline model.
    """
    # 1. Load the dataset (Make sure the path matches your data folder)
    try:
        data_path = "data/pollution_full.csv"
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please check the file name.")
        return

    # 2. Preprocess
    df_clean = preprocess_data(df)

    # 3. Define Features (X) and Target (y)
    # Target: 'pollution' | Features: all other columns
    X = df_clean.drop(columns=['pollution'])
    y = df_clean['pollution']

    # 4. Split data (80% Train, 20% Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Initialize and Train the Baseline Model
    model = LinearRegression()
    print("Training the baseline model (Linear Regression)...")
    model.fit(X_train, y_train)

    # 6. Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("-" * 30)
    print(f"Baseline Results:")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)

if _name_ == "_main_":
    train_baseline()