def preprocess_pollution(df):
    # Remplissage des NaN (très courant dans ce dataset)
    df['pollution'] = df['pollution'].fillna(0)
    # Encodage de la direction du vent (Catégorique -> Numérique)
    df = pd.get_dummies(df, columns=['wnd_dir'])
    return df