import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """
    Load the dataset. Handles cases where 'date' is already combined.
    """
    # Load the raw CSV file
    df = pd.read_csv(filepath)
    
    if 'date' in df.columns:
        print("Detected existing 'date' column. Converting to index...")
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        potential_date_cols = ['year', 'month', 'day', 'hour', 'Year', 'Month', 'Day', 'Hour']
        actual_date_cols = [c for c in potential_date_cols if c in df.columns]
        
        if len(actual_date_cols) >= 4:
            df['date'] = pd.to_datetime(df[actual_date_cols])
            df = df.set_index('date')
            df = df.drop(columns=actual_date_cols)
        else:
            raise KeyError(f"No date information found. Columns: {df.columns.tolist()}")
    
    return df

def preprocess_data(df):
    """
    Clean the data: handle missing values and encode categorical variables.
    """
    # Fill missing values in the target column 'pollution'
    if 'pollution' in df.columns:
        df['pollution'] = df['pollution'].fillna(0)
    
    # Convert categorical wind direction (wnd_dir) into dummy/indicator variables
    if 'wnd_dir' in df.columns:
        df = pd.get_dummies(df, columns=['wnd_dir'])
    
    # Drop the 'No' column if it exists (row identifier)
    if 'No' in df.columns:
        df = df.drop(columns=['No'])
    
    return df

def train_baseline():
    """
    Main pipeline to load, preprocess, train, and evaluate the model.
    """
    print("--- Starting Pipeline ---")
    
    # 1. Loading data
    data_path = "data/pollution_full.csv"
    print(f"Loading data from: {data_path}...")
    try:
        df = load_data(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error during loading: {e}")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    df_clean = preprocess_data(df)

    # 3. Splitting Features and Target
    if 'pollution' not in df_clean.columns:
        print("Error: Target column 'pollution' not found.")
        return
        
    X = df_clean.drop(columns=['pollution'])
    y = df_clean['pollution']

    # 4. Train/Validation Split (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Training
    print("Training Linear Regression baseline...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    print("-" * 30)
    print(f"Baseline Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    train_baseline()