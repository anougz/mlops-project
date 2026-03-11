# Air Pollution Forecasting - MLOps Project 🚀
*A project developed as part of the MLOps course - Winter 2026*

## 👥 Team Members
* **Aurélien Nougarou (@anougz)**
* **Nicolas Barthollet (@nicolasbartho)**

---

## 📝 Project Description
This project implements a complete MLOps pipeline to forecast air pollution levels in Beijing. We utilize a multivariate time series dataset to predict future PM2.5 concentrations based on historical weather and environmental variables.

## 🎯 Task Definition
* **Type of problem**: Time series regression (Multivariate Forecasting).
* **Objective**: Predict pollution levels (PM2.5) at time $t+1$ using data from previous hours.
* **Input features**: Dew point, Temperature, Pressure, Wind direction, Wind speed, Snow, and Rain.

## 📊 Data Source
* **Dataset**: Air Pollution Forecasting - LSTM Multivariate.
* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate/data).
* **Description**: Hourly weather and pollution records for five years from the US Embassy in Beijing, China.

---

## 🏗️ System Architecture
The system is built with a modular architecture to ensure scalability and reproducibility:
1.  **Data Layer**: Raw data ingestion and preprocessing (handling missing values, normalization).
2.  **Experiment Layer**: Training scripts integrated with **MLflow** for tracking.
3.  **Model Registry**: Storage of serialized model artifacts (`.pkl`).
4.  **Serving Layer**: **FastAPI** application containerized with **Docker**.
5.  **CI/CD Layer**: **GitHub Actions** for automated testing and quality enforcement.

---

## ⚙️ MLOps Practices
This project follows industry-standard MLOps principles:
* **Dependency Management**: Powered by `uv` for lightning-fast and deterministic environments.
* **Code Quality**: Automated linting and formatting via **Ruff**.
* **Reproducibility**: Version-controlled environments (`uv.lock`) and containerization (**Docker**).
* **Continuous Integration**: Automated testing suite triggered on every push to ensure code and API stability.
* **Experiment Tracking**: Systematic logging of every training run (hyperparameters, metrics, and models) using **MLflow**.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.12 (Recommended)
* [UV](https://docs.astral.sh/uv/) installed.

### Setup
1. Clone the repo:
```bash
git clone https://github.com/anougz/mlops-project.git
cd mlops-project
```

2. Setup environment and hooks:
```bash
uv sync
uv run pre-commit install
```

---

## 🧪 Testing & Quality Assurance

We use `pytest` for unit testing and `pytest-cov` for coverage analysis.

* **Current Coverage**: 63%
* **Pre-commit Hooks**: Enforces code quality and formatting before every commit.

Run tests:
```bash
uv run pytest --cov=src tests/
```

---

## 📊 Experiment Tracking (MLflow)

Run Training:
```bash
uv run python -m src.train
```

Visualize Results:
```bash
uv run mlflow ui
```

Access the dashboard at http://localhost:5000.

---

## 🔌 API & Deployment

### 1. Serve the API locally:
```powershell
$env:LOCAL_MODEL_PATH="models/your_model.pkl"
uv run python -m uvicorn src.api:app --reload
```

### 2. Docker Support:
```bash
docker build -t pollution-api .
docker run -p 8000:8000 pollution-api
```

---

## 📈 Monitoring & Reliability

Our monitoring strategy focuses on two main axes:

* **Data Drift Detection**: Periodic comparison of incoming feature distributions against training data statistics to identify when the model needs retraining.
* **Health Checks**: The API includes a `/health` endpoint to monitor service uptime and model loading status.
* **Performance Tracking**: Monitoring prediction error (RMSE) in production by comparing forecasts with delayed actual observations.

---

## 🚧 Limitations & Future Work

* **Data Latency**: Currently, the model assumes real-time availability of weather data, which might not be the case in real-world sensors.
* **Advanced Models**: Future work involves implementing Transformer-based architectures or XGBoost to compare against the current baseline.
* **Automated Retraining**: Implementing a "CD" (Continuous Deployment) trigger that retrains the model automatically when data drift is detected.

---

## 📁 Project Structure

* `src/`: Modular source code (data loading, preprocessing, training, API).
* `tests/`: Unit and integration tests.
* `data/`: Local dataset storage.
* `.github/workflows/`: CI/CD pipeline configurations.
* `models/`: Serialized model artifacts.
* `mlruns/`: MLflow experiment metadata.
