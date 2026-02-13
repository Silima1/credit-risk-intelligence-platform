import numpy as np
import pandas as pd

def _to_dataframe(feature_cols, payload: dict):
    # garante ordem e colunas
    row = {c: payload.get(c, None) for c in feature_cols}
    df = pd.DataFrame([row])
    # tenta converter numéricos
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def predict(model, X: pd.DataFrame):
    proba = model.predict_proba(X)
    risk = float(proba[:, 1][0]) if proba.shape[1] > 1 else float(proba[:, 0][0])
    pred = int(model.predict(X)[0])
    return risk, pred