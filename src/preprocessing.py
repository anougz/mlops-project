import pandas as pd

def preprocess_data(df):
    df_clean = df.copy()
    if 'pollution' in df_clean.columns:
        df_clean['pollution'] = df_clean['pollution'].fillna(0)
    if 'wnd_dir' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['wnd_dir'])
    return df_clean
