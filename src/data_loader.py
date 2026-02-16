import pandas as pd

def load_pollution_data(path="data/pollution_full.csv"):
    """
    Loads pollution data and handles date parsing for modern Pandas.
    """
    df = pd.read_csv(path)

    # Handle both pre-combined 'date' or separate columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        date_cols = ['year', 'month', 'day', 'hour']
        # Fallback if columns are capitalized
        if not all(col in df.columns for col in date_cols):
            date_cols = [col.capitalize() for col in date_cols]

        df['date'] = pd.to_datetime(df[date_cols])
        df = df.drop(columns=date_cols)

    df = df.set_index('date')

    if 'No' in df.columns:
        df = df.drop('No', axis=1)

    return df
