import pandas as pd
import numpy as np

def test_full_coverage_run(tmp_path, monkeypatch):
    # 1. Dataset de test (identique)
    d = tmp_path / "data"
    d.mkdir()
    fake_csv = d / "pollution_full.csv"
    df = pd.DataFrame({
        'date': pd.date_range(start='2010-01-01', periods=10, freq='h'),
        'pollution': np.random.rand(10),
        'dew': np.random.rand(10),
        'temp': np.random.rand(10),
        'press': np.random.rand(10),
        'wnd_dir': ['NW', 'SE'] * 5,
        'wnd_spd': np.random.rand(10),
        'snow': [0]*10,
        'rain': [0]*10
    })
    df.to_csv(fake_csv, index=False)

    # Simplify simulation
    class MockMLflow:
        def __init__(self):
            self.sklearn = self
        def set_experiment(self, *args, **kwargs): pass
        def start_run(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def log_param(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def log_model(self, *args, **kwargs): pass

    # Put the mock in the module
    import src.train
    monkeypatch.setattr(src.train, "mlflow", MockMLflow())

    # Mock csv reading to avoid real file
    monkeypatch.setattr("src.data_loader.pd.read_csv", lambda p, **kwargs: df)

    # 3. RUN
    src.train.train_baseline()
    assert True
