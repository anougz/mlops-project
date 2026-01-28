import pandas as pd

def load_pollution_data(path="data/raw_pollution.csv"):
    # Chargement en combinant les colonnes de date
    df = pd.read_csv(path, parse_dates={'date': ['year', 'month', 'day', 'hour']}, index_col='date')
    df.drop('No', axis=1, inplace=True) # On retire l'index inutile
    return df