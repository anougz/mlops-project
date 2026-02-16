import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_pollution_data
from src.preprocessing import preprocess_data

# On définit une variable mlflow pour que monkeypatch puisse la trouver
mlflow = None

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

def train_baseline():
    global mlflow
    if mlflow is None:
        import mlflow as real_mlflow
        import mlflow.sklearn
        mlflow = real_mlflow

    mlflow.set_experiment("Pollution_Prediction")

    with mlflow.start_run(run_name="Linear_Regression_Baseline"):
        print("--- Starting Pipeline ---")

        # 2. Loading data
        data_path = "data/pollution_full.csv"
        print(f"Loading data from: {data_path}...")

        try:
            df = load_pollution_data(data_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"Error during loading: {e}")
            return

        # 3. Preprocessing (L'appel crucial pour ton coverage !)
        print("Preprocessing data...")
        df_clean = preprocess_data(df)

        # 4. Splitting Features and Target
        if 'pollution' not in df_clean.columns:
            print("Error: Target column 'pollution' not found.")
            return

        X = df_clean.drop(columns=['pollution'])
        y = df_clean['pollution']

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 5. Training & MLflow Logging
        print("Training Linear Regression baseline...")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("model_type", "LinearRegression")

        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6. Evaluation & Metrics Logging
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Sauvegarde du modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")

        print("-" * 30)
        print("Baseline Results tracked in MLflow:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print("-" * 30)
        print("Pipeline finished successfully.")

if __name__ == "__main__":
    train_baseline()
