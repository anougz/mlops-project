# MLOps Project Name 🚀
*A project developed as part of the MLOps course - Winter 2026*

## 👥 Team Members
* **Aurélien Nougarou(@anougz)**
* **Nicolas Barthollet (@nicolasbartho)**

---

## 📝 Project Description
This project will focus on a basic MLOps pipeline addressing pollution in Beijing, using the corresponding dataset, and define different machine learning model where we can predict de value of pollution (Y) in function of the others variables (X).

## 🎯 Task Definition
* **Type of problem**: Time series regression (Time Series Forecasting)
* **Objective**:  Predict pollution levels at time t+1 using data from previous hours.
* **Target metric (variables)**: Pollution, Dew, Temperature (temp), Pressure (press), Wind direction (wnd_dir), Wind speed (wnd_spd), Snow (snow), Rain (rain).

## 📊 Data Source (Dataset)
* **Name**: Air Pollution Forecasting - LSTM Multivariate
* **Source**: https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate/data
* **Quick description**:This is a dataset that reports on the weather and the level of pollution each hour for five years at the US embassy in Beijing, China.

The data includes the date-time, the pollution called PM2.5 concentration, and the weather information including dew point, temperature, pressure, wind direction, wind speed and the cumulative number of hours of snow and rain.

---

## 🛠️ Installation & Setup
This project uses **UV** for fast and reproducible dependency management.

### Prerequisites
* Python 3.12+
* [UV](https://docs.astral.sh/uv/) installed on your machine.

### Installation
1. Clone the repo:
```bash
   git clone [https://github.com/anougz/mlops-project.git](https://github.com/your-account/your-repo.git)
   cd your-repo
<<<<<<< Updated upstream
=======
   uv sync
   uv run pre-commit install
```

## 🧪 Testing & Quality Assurance

We use pytest for unit testing and pytest-cov for coverage analysis.

    Current Coverage: 63%

    Linting: Automated with Ruff.

    Pre-commit Hooks: Enforces code quality, formatting, and end-of-file consistency before every commit.

To run tests and see the coverage report:
Bash

uv run pytest --cov=src tests/

## 📊 Experiment Tracking (MLflow)

The training pipeline is fully integrated with MLflow to track hyperparameters (test size, model type) and metrics (RMSE, R2).

    Run Training:
    Bash

    uv run python src/train.py

    Visualize Results:
    Bash

    uv run mlflow ui

    Access the dashboard at http://localhost:5000.

## 📁 Project Structure

    src/: Modular source code (data loading, preprocessing, training).

    tests/: Unit and integration tests.

    data/: Dataset storage.

    .pre-commit-config.yaml: Hooks configuration.
>>>>>>> Stashed changes
